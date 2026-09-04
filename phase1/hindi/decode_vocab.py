import json

def decode_vocab(input_path, output_path):
    print(f"Loading vocabulary from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    readable_vocab = {}
    for key, value in vocab.items():
        # The keys are strings where each character represents a raw byte value
        # We convert each character back to its byte (ord), then decode as utf-8
        byte_arr = bytearray([ord(c) for c in key])
        readable_key = byte_arr.decode('utf-8', errors='replace')
        readable_vocab[readable_key] = value

    print(f"Saving readable vocabulary to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(readable_vocab, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == '__main__':
    decode_vocab('models/hindi_vocabulary.json', 'models/readable_hindi_vocabulary.json')
