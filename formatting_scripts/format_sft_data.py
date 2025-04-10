import re
import json
import sys
import random
from random import shuffle
from argparse import ArgumentParser
import fasttext
fasttext.FastText.eprint = lambda x: None
GLOTLID_PATH = "/scratch/project_462000353/zosaelai2/models/model.bin"
GLOT_MODEL = fasttext.load_model(GLOTLID_PATH)

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--filepath', type=str, help="path to unformatted instruction data")
    ap.add_argument('--outfile', type=str, help="output path")
    ap.add_argument('--lang', default="en", type=str)
    ap.add_argument('--max_samples', default=None, type=int)
    ap.add_argument('--dataset_name', default=None, type=str, help="For Aya only. dataset_name in the Aya collection")
    ap.add_argument('--task_type', default=None, type=str, help="For Aya only. task_type in the Aya collection")
    return ap

def detect_language_glotlid(text):  
    # remove newline from input text
    text = text.replace("\n", " ")
    lab, score = GLOT_MODEL.predict(text)
    lang_code = lab[0].split("__")[-1][:3]
    return lang_code

def format_avoin_avustaja(filepath):
    data = [json.loads(line) for line in open(filepath)]
    

def format_rip_scored_data_for_sft(filepath):
    # data = [json.loads(line) for line in open(filepath)]
    file = open(filepath)
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, "w") as f: 
        for i, line in enumerate(file):
            line = json.loads(line)
            prompt = line['prompt']
            # response = line['best_response']['response']
            response = line['responses'][0]
            lang_prompt = detect_language_glotlid(prompt)
            lang_response = detect_language_glotlid(response)
            if lang_response == lang_prompt:
                messages = {'messages': []}
                messages['messages'].append({'role': 'user', 
                                            'content': prompt})
                messages['messages'].append({'role': 'assistant', 
                                            'content': response})
                f.write(
                    json.dumps(messages, ensure_ascii=False) 
                    + "\n"
                    )
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_openhermes(filepath):
    print("OpenHermes data:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, "w") as f: 
        for line in data:
            messages = {'messages': []}
            conversations = line['conversations']
            for turn in conversations:
                if turn['from'] == 'human':
                    messages['messages'].append({'role': 'user', 
                                             'content': turn['value']})
                elif turn['from'] == 'gpt':
                    messages['messages'].append({'role': 'assistant', 
                                             'content': turn['value']})
            # add metadata
            # messages['source'] = line['source']
            # messages['system_prompt'] = line['system_prompt']
            # messages['custom_instruction'] = line['custom_instruction']
            # messages['category'] = line['category']
            f.write(
                json.dumps(messages, ensure_ascii=False) 
                + "\n"
                )
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_capybara(filepath):
    print("Capybara data:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, "w") as f: 
        for line in data:
            messages = {'messages': []}
            conversations = line['conversation']
            for turn in conversations:
                messages['messages'].append({'role': 'user',
                                             'content': turn['input']})
                messages['messages'].append({'role': 'assistant',
                                             'content': turn['output']})
            # messages['source'] = line['source']
            f.write(json.dumps(messages, ensure_ascii=False) 
                    + "\n"
                    )
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_dolly(filepath, lang="en"):
    print("Dolly data:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    if lang == "en":
        context_col = "orig_context"
        instruction_col = "orig_instruction"
        response_col = "orig_response"
    else:
        context_col = "context"
        instruction_col = "instruction"
        response_col = "response"   
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, "w") as f: 
        for line in data:
            messages = {'messages': []}
            if not line[context_col] or line[context_col].isspace():
                user_message = {'role': 'user',
                                'content': line[instruction_col]
                                }
            else:
                user_message = {'role': 'user',
                                'content': line[context_col] + "\n" + line[instruction_col]
                                }
            assistant_message = {'role': 'assistant',
                                 'content': line[response_col]
                                }
            messages['messages'].append(user_message)
            messages['messages'].append(assistant_message)
            f.write(json.dumps(messages , ensure_ascii=False) +
                    "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_lima(filepath):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for i, line in enumerate(data):
            messages = {'messages':[]}
            conversations = line['conversations']
            if len(conversations) > 1:
                messages['messages'].append({'role': 'user',
                                             'content': conversations[0]})
                messages['messages'].append({'role': 'assistant',
                                             'content': conversations[1]})
                f.write(json.dumps(messages, ensure_ascii=False) + 
                        "\n")
            else:
                print("Line", i, "missing some turns")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_oasst1_octopack(filepath, lang='en'):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for line in data:
            if line['lang'] == lang:
                messages = {'messages': []}
                conversation = line['conversations']
                for turn in conversation:
                    content = turn['text']
                    role = turn['role']
                    if role == 'prompter':
                        role = 'user'
                    messages['messages'].append({'role': role,
                                                 'content': content
                                                 })
                f.write(json.dumps(messages, ensure_ascii=False) + 
                        "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_oasst2_curated(filepath):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for line in data:
            messages = {'messages': []}
            conversation = line['messages']
            for turn in conversation:
                messages['messages'].append({'role': turn['role'],
                                             'content': turn['content']})
            f.write(json.dumps(messages, ensure_ascii=False) + 
                    "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)


def format_hh_rlhf(filepath):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for entry in data:
            messages = {'messages': [] }
            conversation = entry['chosen']
            split_converstion = re.split("\n\nHuman: |\n\nAssistant: ", conversation)[1:]
            for t, turn in enumerate(split_converstion):
                if len(turn) > 0:
                    if t%2 == 0:
                        messages['messages'].append({'role': 'user',
                                                     'content': turn})
                    else:
                        messages['messages'].append({'role': 'assistant',
                                                     'content': turn})
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_glaive_code_assistant(filepath, max_samples=None, shuffle_data=True):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    if shuffle_data:    
        shuffle(data)
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for i, line in enumerate(data):
            if max_samples and i >= max_samples:
                break
            messages = {'messages':[]}
            messages['messages'].append({'role': 'user',
                                         'content': line['question']})
            messages['messages'].append({'role': 'assistant',
                                         'content': line['answer']})
            f.write(json.dumps(messages, ensure_ascii=False) + 
                    "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_argilla_10k_prompts_ranked(filepath, human_only=True):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for i, line in enumerate(data):
            messages = {'messages':[]}
            if (human_only is False) or (human_only is True and line['kind']) == 'human':
                user_content = line['generation_prompt'][0][1]['content']
                assistant_content = line['generations'][0]
                messages['messages'].append({"role": "user",
                                             "content": user_content})
                messages['messages'].append({"role": "assistant",
                                         "content": assistant_content})
                f.write(json.dumps(messages, ensure_ascii=False) + 
                        "\n")
    print("Done! Saved SFTTrainer-formatted data as", outfile)


def format_cosmopedia(filepath, filter_synthetic=True, max_length=1000):
    synthetic_sources = ['openhermes2.5', 'ultrachat']
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for entry in data:
            if filter_synthetic is True and entry['seed_data'] not in synthetic_sources and entry['text_token_length'] < max_length:
                messages = {'messages':[]}
                messages['messages'].append({"role": "user",
                                            "content": entry['prompt']})
                messages['messages'].append({"role": "assistant",
                                            "content": entry['text']})
                f.write(json.dumps(messages, ensure_ascii=False) + 
                        "\n")
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_dolly_megatron_finetuning(filepath):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-megatron.jsonl")
    with open(outfile, 'a') as f:
        for entry in data:
            text = {"text": "<user> " + entry['messages'][0]['content'] + " <assistant> " + entry['messages'][1]['content']}
            f.write(json.dumps(text, ensure_ascii=False) + "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_daring_anteater(filepath):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for entry in data:
            messages = {'messages': []}
            conversation = entry['conversations']
            for turn in conversation:
                messages['messages'].append({'role': turn['from'].lower(),
                                             'content': turn['value']})
            f.write(json.dumps(messages, ensure_ascii=False) + 
                    "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_open_platypus(filepath):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for entry in data:
            messages = {'messages': []}
            prompt = entry['instruction']
            response = entry['output']
            messages['messages'].append({'role': 'user',
                                        'content': prompt
                                        })
            messages['messages'].append({'role': 'assistant',
                                        'content': response
                                        })
            f.write(json.dumps(messages, ensure_ascii=False) + 
                    "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_web_instruct(filepath):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for entry in data:
            messages = {'messages': []}
            prompt = entry['orig_question']
            response = entry['orig_answer']
            messages['messages'].append({'role': 'user',
                                            'content': prompt
                                        })
            messages['messages'].append({'role': 'assistant',
                                            'content': response
                                        })
            f.write(json.dumps(messages, ensure_ascii=False) + 
                    "\n")
        f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_magpie(filepath, multi_turn=False):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        if multi_turn:
            for entry in data:
                messages = {'messages': []}
                for key, value in entry.items():
                    if "prompt" in key:
                        messages['messages'].append({'role': 'user',
                                                    'content': value
                                                    })
                    elif "response" in key:
                        messages['messages'].append({'role': 'assistant',
                                                    'content': value
                                                    })
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
        else:
            for entry in data:
                messages = {'messages': []}
                messages['messages'].append({'role': 'user',
                                         'content': entry['prompt']
                                         })
                messages['messages'].append({'role': 'assistant',
                                            'content': entry['response']
                                        })        
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
    f.close()
    print("Done! Saved SFTTrainer-formatted data as", outfile)

def format_sdsd_dialogues(filepath, add_system_role=False):
    print("filepath:", filepath)
    data = [json.loads(line) for line in open(filepath)]
    outfile = filepath.replace(".jsonl", "-sfttrainer.jsonl")
    with open(outfile, 'a') as f:
        for entry in data:
            if entry['is_violation'] == False:
                valid_entry = True
                messages = {'messages': []}
                for turn in entry['messages']:
                    if "ASSISTANT" in turn['content']:
                        valid_entry = False
                        break
                    else:
                        if turn['role'] != "system" or (turn['role'] == "system" and add_system_role):
                            messages['messages'].append({'role': turn['role'],
                                                            'content': turn['content']
                                                        })
                if valid_entry and len(messages['messages']) > 0:
                    f.write(json.dumps(messages, ensure_ascii=False) + "\n")

def format_aya(filepath, outfile, max_samples, dataset_name, task_type):
    if dataset_name is not None:
        print("Formatting Aya with dataset_name:", dataset_name)
    if task_type is not None:
        print("Formatting Aya with task_type:", task_type)
    count = 0
    file = open(filepath)
    with open(outfile, "w") as f:
        for i, line in enumerate(file):
            if max_samples is None or i < max_samples:
                entry = json.loads(line)
                if (dataset_name is not None and entry['dataset_name']==dataset_name) or (task_type is not None and entry['task_type']==task_type):
                    messages = {'messages': []}
                    messages['messages'].append(
                        {
                            'role': 'user',
                            'content': entry['inputs']
                        }
                    )
                    messages['messages'].append(
                        {
                            'role': 'assistant',
                            'content': entry['targets']
                        }
                    )
                    count +=1
                    f.write(json.dumps(messages, ensure_ascii=False) + "\n")
    print(f"Total: {count}")

def format_rip_dpo_data(filepath, outfile):
    file = open(filepath)
    with open(outfile, 'w') as f:
        for i, line in enumerate(file):
            entry = json.loads(line)
            prompt = entry['prompt']
            best_response = entry['best_response']['response']
            best_score = entry['best_response']['score']
            # for rejected, pick a random response with score < best_score
            random_index = random.choice(range(len(entry['all_responses'])))
            random_score = entry['all_responses'][random_index]['score']
            while random_score > best_score:
                random_index = random.choice(range(len(entry['all_responses'])))
                random_score = entry['all_responses'][random_index]['score'] 
            random_response = entry['all_responses'][random_index]['response']
            chosen = [{'role': 'user', 
                       'content': prompt},
                       {'role': 'assistant',
                        'content': best_response}
            ]
            rejected = [{'role': 'user',
                         'content': prompt
                        },
                        {'role': 'assistant',
                         'content':random_response
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
    lang = args.lang
    max_samples = args.max_samples
    if "hermes" in filepath.lower():
        format_openhermes(filepath)
    elif "capybara" in filepath.lower():
        format_capybara(filepath)
    elif "dolly" in filepath.lower():
        format_dolly(filepath, lang)
    elif "lima" in filepath.lower():
        format_lima(filepath)
    elif "oasst1" in filepath.lower():
        format_oasst1_octopack(filepath, lang='en')
    elif "oasst2" in filepath.lower():
        format_oasst2_curated(filepath)
    elif "hh_rlhf" in filepath.lower() or "hh-rlhf" in filepath.lower():
        format_hh_rlhf(filepath)
    elif "glaive" in filepath.lower():
        format_glaive_code_assistant(filepath, max_samples)
    elif "10k_prompts_ranked" in filepath.lower():
        format_argilla_10k_prompts_ranked(filepath)
    elif "cosmopedia" in filepath.lower():
        format_cosmopedia(filepath)
    elif "daring-anteater" in filepath.lower():
        format_daring_anteater(filepath)
    elif "open-platypus" in filepath.lower():
        format_open_platypus(filepath)
    elif "webinstruct" in filepath.lower():
        format_web_instruct(filepath)
    elif "magpie" in filepath.lower():
        if "single" in filepath.lower():
            format_magpie(filepath)
        else:
            format_magpie(filepath, multi_turn=True)
    elif "sdsd" in filepath.lower():
        format_sdsd_dialogues(filepath, add_system_role=False)
    elif "aya" in filepath.lower():
        format_aya(filepath, args.outfile, args.max_samples, args.dataset_name, args.task_type)
    elif "rip" in filepath.lower() or "tulu" in filepath.lower():
        format_rip_scored_data_for_sft(filepath)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
