class Hyperparams:

	source_file = 'en-chars.txt'
	target_file = 'div-chars.txt'
	training_file = 'transdata.txt'
	test_file  = 'testdata.txt'

	hidden = 128
	max_seq_len = 35
	epoch = 30
	batch = 128
	char_emb_dim = 64

	model_name = 'en-div-transliteration'
