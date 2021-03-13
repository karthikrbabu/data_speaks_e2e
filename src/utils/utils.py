# Utilities for processing model outputs, and writing it out for later usage

import tensorflow as tf
import time


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
        print()
        print("Took %.2f seconds" % ((end - start)))
        print("Took {0} minutes".format((end - start)/60))
        print()
        return output

    
    train_output = gen_output(tf_train_ds, "Train", gen_params) if tf_train_ds else []
    validation_output = gen_output(tf_valid_ds, "Validation", gen_params) if tf_valid_ds else []
    test_output = gen_output(tf_test_ds, "Test", gen_params) if tf_test_ds else []
    
    return {"train_output": train_output, "validation_output":validation_output, "test_output":test_output}




def write_out_tsv(hf_ds, hf_ds_name, model_out, write_path='./'):
    """
    Write out TSV file once we have gotten respective datasets model generated outputs
    
    hf_ds => HuggingFaceDataset - features: ['meaning_representation', 'human_reference']
    gen_out => List - Corresponding model generated outputs for hf_ds
    hf_ds_name => String - Name of Dataset
    
    """

    print(f"Writing {hf_ds_name}_out.csv >> {write_path}")
    source = hf_ds['meaning_representation']
    reference = hf_ds['human_reference']
    system_output = model_out
    df = pd.DataFrame({"source": source, "reference":reference, "output":system_output})
    
    df.to_csv(f'{write_path}/{hf_ds_name}_out.tsv', sep='\t', header=True, index=False)




def encode(example, tokenizer, encoder_max_len=60, decoder_max_len=60):
    """
    Encode function that uses the T5 Tokenizer on each example
    """
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



def print_kwargs(**kwargs):
    # Iterating over the Python kwargs dictionary
    for k, v in kwargs.items():
        print(k, v)


