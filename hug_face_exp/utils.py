#Utilities for processing model outputs, and writing it out for later usage

def get_model_output(model, tok, tf_train_ds=None, tf_valid_ds=None, tf_test_ds=None):
    """
    Once model is trained and saved. Use this function to evaluate the output on train, validation, and test 
    
    model => TFPretrainedModel - T5Wrapper base class
    tok => AutoTokenizer - Tokenizer used for the respective model
    tf_train_ds => PreFetchDataset - of Batched Tensors (train)
    tf_valid_ds => PreFetchDataset - of Batched Tensors (validation)
    tf_test_ds => PreFetchDataset - of Batched Tensors (test)
    
    """
    
    def gen_output(ds, ds_name):
        """
        ds => PreFetchDataset  - of Batched Tensors
        ds_name => String - Name of Dataset
        
        """
        
        print(f"Starting {ds_name}")
        start = time.time()
        output = []
        ds_iter = iter(list(ds))
        
        isNext = True
        while isNext:
            input_batch = next(ds_iter, None)
            if input_batch:
                input_batch.pop('labels', None)
                input_batch.pop('decoder_attention_mask', None)

                hypotheses_batch = model.generate(
                    **input_batch,
                    num_beams=4,
                    length_penalty=2.0,
                    max_length=142,
                    min_length=56,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    early_stopping=True,
                )
                decoded = tok.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                output += decoded
            else:
                isNext = False

        timestamp2 = time.time()
        print("Took %.2f seconds" % ((timestamp2 - timestamp1)))
        print("Took {0} minutes".format((timestamp2 - timestamp1)/60))
        print()
        return output

    
    train_output = gen_output(tf_train_ds, "Train") if tf_train_ds else []
    validation_output = gen_output(tf_valid_ds, "Validation") if tf_valid_ds else []
    test_output = gen_output(tf_test_ds, "Test") if tf_test_ds else []
    
    return {"train_output": train_output, "validation_output":validation_output, "test_output":test_output}
    

    
    
def write_out_tsv(hf_ds, hf_ds_name, sys_out):
    """
    Write out TSV file once we have gotten respective datasets model generated outputs
    
    hf_ds => HuggingFaceDataset - features: ['meaning_representation', 'human_reference']
    gen_out => List - Corresponding model generated outputs for hf_ds
    hf_ds_name => String - Name of Dataset
    
    """

    print(f"Writing {hf_ds_name}_out.csv")
    source = hf_ds['meaning_representation']
    reference = hf_ds['human_reference']
    system_output = sys_out
    df = pd.DataFrame({"source": source, "reference":reference, "output":system_output})
    
    df.to_csv(f'{hf_ds_name}_out.tsv', sep='\t', header=True, index=False)
    