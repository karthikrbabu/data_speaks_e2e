# Utilities for processing model outputs, and writing it out for later usage

# +
import tensorflow as tf
import time
import os.path
import os
import subprocess
import csv

#AWS
import boto3
s3 = boto3.resource('s3')


# +
def get_model_output(model, tok, gen_params, tf_train_ds=None, tf_valid_ds=None, tf_test_ds=None):
    """
    Once model is trained and saved. Use this function to evaluate the output on train, validation, and test 
    
    model => TFPretrainedModel - T5Wrapper base class
    tok => AutoTokenizer - Tokenizer used for the respective model
    tf_train_ds => PreFetchDataset - of Batched Tensors (train)
    tf_valid_ds => PreFetchDataset - of Batched Tensors (validation)
    tf_test_ds => PreFetchDataset - of Batched Tensors (test)
    **kwargs => Provided to model.generate function call
    
    """

    def gen_output(ds, ds_name, gen_params):
        """
        ds => PreFetchDataset  - of Batched Tensors
        ds_name => String - Name of Dataset
        
        """
        print(f"Starting {ds_name}:")
        print()
        print_kwargs(**gen_params)
        
        start = time.time()
        output = []
        tf_list = list(ds)
        total_batches = len(tf_list)
        print("total_batches: ", total_batches)
        ds_iter = iter(tf_list)
        
        iterator_built_time = time.time()
        print("Built Iterator %.2f seconds" % ((iterator_built_time - start)))
        print("Built Iterator {0} minutes".format((iterator_built_time - start)/60))
        print()
        
        batch_num = 1
        
        isNext = True
        while isNext:
            input_batch = next(ds_iter, None)
            if input_batch:
                batch_start = time.time()
                print(f"start batch_num {batch_num}/{total_batches} - ", end='')
                input_batch.pop('labels', None)
                input_batch.pop('decoder_attention_mask', None)

                hypotheses_batch = model.generate(
                    **input_batch,
                    **gen_params
                )
                decoded = tok.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                output += decoded
                batch_end = time.time()
                print("done %.2f seconds" % ((batch_end - batch_start)))
                batch_num += 1
            else:
                isNext = False

        end = time.time()
        tot_time_min = (end - start)/60
        print()
        print("Took %.2f seconds" % ((end - start)))
        print("Took {0} minutes".format(tot_time_min))
        print()
        
        return output

    
    train_output = gen_output(tf_train_ds, "Train", gen_params) if tf_train_ds else []
    validation_output = gen_output(tf_valid_ds, "Validation", gen_params) if tf_valid_ds else []
    test_output = gen_output(tf_test_ds, "Test", gen_params) if tf_test_ds else []
        
    return {"train": {"output": train_output}, "validation": {"output": validation_output}, "test":{"output": test_output}, "gen_params": gen_params}



# -




def write_model_output(hf_ds, hf_ds_name, ts, model_out, write_path='.'):
    """
    Write out two files one with the provided references, one with the model output references
    
    ts=> timestamp
    hf_ds => HuggingFaceDataset - features: ['meaning_representation', 'human_reference']
    gen_out => List - Corresponding model generated outputs for hf_ds
    hf_ds_name => String - Name of Dataset
    
    """

    path = f'{write_path}/output/{ts}'
    if not os.path.exists(f'{path}'):
        os.makedirs(path)
    else:
        print("removing: path")
        os.remove(path)
          
    ref_file_path = f'{path}/{hf_ds_name}_ref.txt'
    ref_model_file_path = f'{path}/{hf_ds_name}_ref_model.txt'

    print(f"Writing {hf_ds_name} files >> {path}")
    reference = hf_ds['human_reference']
    system_output = model_out
    
    with open(ref_file_path, 'a') as ref_ds:
        for ref in list(reference):
            ref_ds.write(f'{ref}\n')

    with open(ref_model_file_path, 'a') as ref_model:
        for ref in system_output:
            ref_model.write(f'{ref}\n')

    print("Wrote: ", ref_file_path)
    print("Wrote: ", ref_model_file_path)


def compute_metrics(output_files_path, base_dir, ts, ds_name, gen_params):
    """
    output_files_path => provide path to access ref + model_ref txt files
    ds_name => name of data i.e. ['train', 'validation', 'test']
    gen_params => parameters used for model
    """
    
    #Prepare the files paths
    model_version = ts
    ref_file_path = f'{output_files_path}/output/{ts}/{ds_name}_ref.txt'
    ref_model_file_path = f'{output_files_path}/output/{ts}/{ds_name}_ref_model.txt'
    
    print("ref_file_path: ", ref_file_path)
    print("ref_model_file_path: ", ref_model_file_path)
    
    #Build the script command, execute with Popen
    cmd = f"""{base_dir}/src/metrics_script/measure_scores.py {ref_file_path} {ref_model_file_path} -p -t -H"""
    p = subprocess.Popen(f'{cmd}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    outs = []
    for line in p.stdout.readlines():
        outs.append(line)
    retval = p.wait()
    
    #Get expected output in dictionary format
    headers = outs[-2].decode("utf-8").strip().split()
    vals = outs[-1].decode("utf-8").strip().split()
    output = dict(zip(headers, vals))
    output['version'] = 'v' + model_version
    output['params'] = gen_params
    return output


def save_metrics(exp_dir, ts, metric_scores):
    """
    Save the metrics/metadat for this run
    """
    file_name = f"{exp_dir}/output/{ts}/metrics.csv"
    line = [metric_scores['version'], metric_scores['BLEU'], metric_scores['NIST'], \
            metric_scores['METEOR'], metric_scores['ROUGE_L'], metric_scores['CIDEr'], \
            metric_scores['params'], metric_scores['File']]

    header = ['version', 'BLEU', 'NIST', 'METEOR', 'ROUGE_L', 'CIDEr', 'params', 'File']

    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header) # write the header
        
        # write the actual content line by line
        writer.writerow(line)
    print("Added Record")



def encode(example, tokenizer, is_custom=False, encoder_max_len=60, decoder_max_len=60):
    """
    Encode function that uses the T5 Tokenizer on each example
    """
    if is_custom:
        mr = example[0]
        ref = example[1]
    else:    
        mr = example['meaning_representation']
        ref = example['human_reference']
  
    mr_base = f"data_to_text: {str(mr)}"
    ref_base = f"{str(ref)}"

    encoder_inputs = tokenizer(mr_base, truncation=True, 
                               return_tensors='tf', max_length=encoder_max_len,
                              pad_to_max_length=True)

    decoder_inputs = tokenizer(ref_base, truncation=True, 
                               return_tensors='tf', max_length=decoder_max_len,
                              pad_to_max_length=True)
    
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               'labels':target_ids, 'decoder_attention_mask':target_attention}
    return outputs



def to_tf_dataset(dataset):
    """
    Encode_Tf function that applies our custom encode function on each tensor example
    """
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    dataset.set_format(type='tensorflow', columns=columns)
    return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 
                'labels':tf.int32, 'decoder_attention_mask':tf.int32}
    return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
                  'labels': tf.TensorShape([None]), 'decoder_attention_mask':tf.TensorShape([None])}
    ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
    return ds



def create_dataset(dataset, cache_path=None, batch_size=30, 
                   buffer_size= 1000, shuffling=True):
    """
    Builds data object ready for use given our training dataset in the form of tensors
    """

    if cache_path is not None:
        dataset = dataset.cache(cache_path)        
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



def save_model_to_s3(model, model_path, best=False):
    """
    Saves the trained model to a local directory and then upload it to s3
    In s3, it first clears the exisiting 'latest' model and
        uploads the new model to latest and also by its timestamp
    """

    model.save_pretrained(f'{model_path}')
    s3_bucket=s3.Bucket('w266-karthik-praveen')
    for obj in s3_bucket.objects.filter(Prefix='latest/'):
        s3.Object(s3_bucket.name,obj.key).delete()
    s3_base=model_path.split("/")[-2]
    s3_bucket.upload_file(f'{model_path}/config.json',f'{s3_base}/config.json')
    s3_bucket.upload_file(f'{model_path}/tf_model.h5',f'{s3_base}/tf_model.h5')
    if(best): 
        s3_bucket.upload_file(f'{model_path}/config.json',f'best/{s3_base}/config.json')
        s3_bucket.upload_file(f'{model_path}/tf_model.h5',f'best/{s3_base}/tf_model.h5')



def print_kwargs(**kwargs):
    # Iterating over the Python kwargs dictionary
    for k, v in kwargs.items():
        print(k, v)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


