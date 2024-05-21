# additional Python packages for S2T data processing/model training
# pip install pandas torchaudio sentencepiece

python examples/speech_to_text/prep_librispeech_data.py \
  --output-root /DB --vocab-type unigram --vocab-size 300