import re
import json
import sys
import random
from random import shuffle
from argparse import ArgumentParser

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--filepath', type=str, help="path to unformatted instruction data")
    ap.add_argument('--outfile', type=str, help="output path")
    ap.add_argument('--lang', default="en", type=str)
    ap.add_argument('--max_samples', default=None, type=int)
    return ap

def format_rip_dpo_data(filepath, outfile, use_random_rejected_response=False):
    print("Formatting RIP data for DPO")
    print("Original data:", filepath)
    print("Output:", outfile)
    file = open(filepath)
    with open(outfile, 'w') as f:
        for i, line in enumerate(file):
            entry = json.loads(line)
            prompt = entry['prompt']
            best_response = entry['best_response']['response']
            best_score = entry['best_response']['score']
            chosen = [{'role': 'user', 
                       'content': prompt},
                       {'role': 'assistant',
                        'content': best_response}
            ]
            if use_random_rejected_response:
                # for rejected, pick a random response with score < best_score
                random_index = random.choice(range(len(entry['all_responses'])))
                random_score = entry['all_responses'][random_index]['score']
                while random_score > best_score:
                    random_index = random.choice(range(len(entry['all_responses'])))
                    random_score = entry['all_responses'][random_index]['score'] 
                random_response = entry['all_responses'][random_index]['response']
                rejected = [{'role': 'user',
                            'content': prompt
                            },
                            {'role': 'assistant',
                            'content':random_response
                            }
                        ]
            else:
                worst_response = entry['worst_response']['response']
                rejected = [{'role': 'user',
                             'content': prompt
                            },
                            {'role': 'assistant',
                            'content':worst_response
                            }
                        ]

            dpo_entry = {
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            }
            f.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")
    print("Done! Saved DPO data to", outfile)



def main(argv):
    args = argparser().parse_args(argv[1:])
    filepath = args.filepath
    outfile = args.outfile
    lang = args.lang
    max_samples = args.max_samples
    if "rip" in filepath.lower():
        format_rip_dpo_data(filepath, outfile)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
