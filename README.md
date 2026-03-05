# Silent-Speech-EMG-NeuroSymbolic-LLM
EMG silent-speech recognition using a trainable EMG adapter to condition a frozen decoder-only LLM, trained with AR+CTC and decoded with NeuroSymbolic constrained beam search (trie + 5-gram fusion + boundary/EOS control) with optional CTC reranking/adaptive fusion.
