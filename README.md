# NLP-Role-Labeler
Semantic Role Labeler in Natural Language Processing

    Usage:
        Put file 'semantic_role_labeler.py' and folder 'data.wsj' in the same folder.
        -semantic_role_labeler.py
        -data.wsj
          |---ne                                  # ne : Named Entities.
          |---props                               # props : Target verbs and correct propositional arguments.
          |---synt.cha                            # synt.cha : PoS tags and full parses of Charniak.
          |---words                               # words : words.
        -data
          |---test-set.txt                        # get from .sh file.
          |---train-set.txt                       # get from .sh file.
        -temp
          |---GoogleNews-vectors-negative300.bin  # embedding file.
        -models
          |---model.pth                           # best model we get.
        -outputs
          |---outputs.txt                         # model outputs.
          |---test_outputs.txt                    # outputs that satisfies HW requirement.
        -make-testset.sh                          # run with bash to get test set.
        -make-trainset.sh                         # run with bash to get train set.
        -senmantic_role_labeler.txt               # log file.
        -srl-eval.pl
    Command to run:
        python semantic_role_labeler.py
    Description:
        Build and train a recurrent neural network (RNN) with hidden vector size 256.
        Loss function: Adam loss.
        Embedding vector: 300-dimensional.
        Learning rate: 0.0001.
        Batch size: 16
