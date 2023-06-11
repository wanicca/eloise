import os
import copy
import sys
from enum import Enum
import time
import types
import gc
import re
import numpy as np
import torch
import pickle
import translate
import langid

# '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_JIT_ON"] = '1'
# '1' : use CUDA kernel for seq mode (much faster)
os.environ["RWKV_CUDA_ON"] = '1'


from model.model_run import RWKV
from model.utils import TOKENIZER, SAMPLER
from prompt import User, Scenario, SCENARIO_ASSISTANT, SCENARIO_CHAT

import prompt

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)


CHAT_LANG = 'English'  # English Chinese
# CHAT_LANG = 'Chinese'
SAME_LANG = "PLEASE SELECT TWO DISTINCT LANGUAGES"

#tokenizer = TOKENIZER("20B_tokenizer.json")
tokenizer = TOKENIZER("rwkv_vocab_v20230424")

DONT_OUTPUT = -float('inf')
END_OF_TEXT = 0
END_OF_LINE = 187
END_OF_LINE_DOUBLE = 535

MAX_MESSAGE_LEN = 8192
CHUNK_LEN = 256

MAX_GENERATE_LEN = 250
MAX_REPLY_LEN = 1024
MAX_USER_ARCHIVE = 10

CHAT_SAMPLER = SAMPLER("typical", 1.2, 0.6, 0.8, 0.8, 0.4, 0.4, 256)
INSTRUCT_SAMPLER = SAMPLER("nucleus", 0.8, 1.0, 0.5, 0.95, 0.1, 0.1, 256)

args = types.SimpleNamespace()

# args.strategy = 'cpu fp32'
# args.strategy = 'cuda fp16'
# args.strategy = 'cuda fp16 *8 -> cpu fp32'
# args.strategy = 'cuda fp16 *6+'
# args.strategy = 'cuda fp16 *0+ -> cpu fp32 *1'
#args.strategy = 'cuda fp16i8 *16 -> cuda fp16'
# args.strategy = 'cuda fp16 *20 -> cpu fp32'
args.strategy = 'cuda fp16'

#args.MODEL_NAME = "/root/autodl-tmp/rwkv-raven7bv10x-sblend-0426-v2-4096-epoch13"
#args.MODEL_NAME = '/root/autodl-tmp/world/RWKV-4-World-7B-v1-OnlyForTest_52%_trained-20230606-ctx4096'
args.MODEL_NAME = '/root/autodl-tmp/world/RWKV-4-World-7B-v1-OnlyForTest_64%_trained-20230610-ctx4096'

#args.STATE_DUMP_NAME = './state_14b'
#args.STATE_DUMP_NAME = './state_7b'
args.STATE_DUMP_NAME = './state_7b_world'
# args.STATE_DUMP_NAME = 'states/7b.state'
args.INIT_REMOVE_PRECOMPUTED_STATE = True #载入模型时删除之前预计算的STATE
args.MAX_PRECOMPUTED_STATE = 100 #保存的预计算STATE数量，超出则删除

# args.LORA_NAME = "/root/autodl-tmp/rwkv-raven14bv11x-lora-0526-ffn-4096-epoch3.pth"
# args.LORA_NAME = "/root/autodl-tmp/rwkv-raven-14B-lora-mix2.pth"
# args.LORA_NAME = "/root/autodl-tmp/rwkv-raven7bv10x-lora-0426-v2-4096-epoch13.pth" #巧兹
#args.LORA_NAME ="/root/autodl-tmp/rwkv-raven7b-sblend-augmented-only.pth"
#args.LORA_NAME ="/root/autodl-tmp/world/rwkv-5-600.pth"
#args.LORA_NAME = "/root/TrainChatGalRWKV/RWKV-v4neo-lora/out7b_lora-0611_r16_b32_c4096_1e-5/rwkv-5-600.pth"
#args.LORA_NAME = "/root/TrainChatGalRWKV/RWKV-v4neo-lora/out7b_lora-world-64/rwkv-5-600.pth"
args.LORA_NAME = "/root/TrainChatGalRWKV/RWKV-v4neo-lora/out7b_lora-world-64-14-31/rwkv-9-1000.pth"

# 10 12 10 = 32
# args.LORA_FILTER = '0.0*0-23 0.8*24-31'
# args.LORA_FILTER = '0-39'
args.LORA_FILTER = '0.076*0 0.091*1 0.108*2 0.128*3 0.151*4 0.178*5 0.208*6 0.241*7 0.279*8 0.319*9 0.363*10 0.408*11 0.456*12 0.504*13 0.552*14 0.599*15 0.645*16 0.688*17 0.728*18 0.764*19 0.798*20 0.827*21 0.853*22 0.876*23 0.895*24 0.912*25 0.926*26 0.939*27 0.949*28 0.957*29 0.965*30 0.971*31'

import json
with open('./lora_cfg.json', 'r') as openfile:
    lora_cfg = json.load(openfile)
    args.LORA_FILTER = lora_cfg['lora_cfg']

class GenerateMode(Enum):
    GENERATE = 0
    INSTRUCT = 1
    RETRY = 2
    MORE = 3


# Load Model
print(f"Loading... {args.MODEL_NAME}")
# os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy, lora=args.LORA_NAME,lora_alpha=32,lora_layer_filter=args.LORA_FILTER)


def run_rnn(tokens, model_state=None):
    tokens = [int(x) for x in tokens]

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out, model_state


def state_to_cuda(state):
    if state:
        for i in range(model.args.n_layer):
            dd = model.strategy[i]
            dev = dd.device
            state[i*5+0] = state[i*5+0].to(dev)
            state[i*5+1] = state[i*5+1].to(dev)
            state[i*5+2] = state[i*5+2].to(dev)
            state[i*5+3] = state[i*5+3].to(dev)
            state[i*5+4] = state[i*5+4].to(dev)


def state_to_cpu(state):
    if state:
        for i in range(model.args.n_layer):
            state[i*5+0] = state[i*5+0].cpu()
            state[i*5+1] = state[i*5+1].cpu()
            state[i*5+2] = state[i*5+2].cpu()
            state[i*5+3] = state[i*5+3].cpu()
            state[i*5+4] = state[i*5+4].cpu()


all_state = {}


def clean_user_state(uid, channel):
    n = f'{uid}_{channel}'
    if n in all_state.keys():
        del all_state[n]


def save_all_state(uid, channel, last_out, model_state, model_tokens):
    n = f'{uid}_{channel}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['state'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)
    state_to_cpu(all_state[n]['state'])


def load_all_state(uid, channel):
    n = f'{uid}_{channel}'
    model_state = copy.deepcopy(all_state[n]['state'])
    model_tokens = copy.deepcopy(all_state[n]['token'])

    state_to_cuda(model_state)
    return all_state[n]['out'], model_state, model_tokens


def save_params(uid, channel, **kwargs):
    n = f'params_{uid}_{channel}'
    all_state[n] = kwargs


def load_params(uid, channel):
    n = f'params_{uid}_{channel}'
    return all_state[n]

def copy_state(s_uid,s_channel,t_uid,t_channel):
    last_out, model_state, model_tokens = load_all_state(s_uid,s_channel)
    save_all_state(t_uid,t_channel,last_out, model_state, model_tokens)

def copy_params(s_uid,s_channel,t_uid,t_channel):
    params = load_params(s_uid,s_channel)
    save_params(t_uid,t_channel,**params)

def fix_params_mismatch(params,channel,extra_params=''):
    """
    add default params when channel mismatched.
    """
    if channel == 'gen':
        if 'mode' not in params:
            params['mode'] = GenerateMode.GENERATE
    elif channel == 'chat':
        if 'scenario' not in params or extra_params:
            ## move parse process into Scenario
            # kwargs = {}
            # if channel == 'chat':
            #     kwargs['user_name'] = re.search(r'-u(?:ser)?(?:\=|\s+)(\S+)',extra_params).group(1)
            #     kwargs['bot_name'] = re.search(r'-b(?:ot)?(?:\=|\s+)(\S+)',extra_params).group(1)
            #     kwargs['intro'] = 'chat_intro'
            # params['scenario'] = Scenario(**kwargs)
            scenario = params.get('scenario') or SCENARIO_CHAT
            scenario = copy.deepcopy(scenario)
            scenario.parse(extra_params)
            params['scenario'] = scenario
    params['sampler'].parse(extra_params) #Setting new sampler params
    

def copy_all(s_uid,s_channel,t_uid,t_channel,extra_params=""):
    """
    only support gen and chat
    """
    if s_channel == 'gen':
        s_channel_now = 'gen_1'
        s_channel_pre = 'gen_0'
    elif s_channel == 'chat':
        s_channel_now = 'chat'
        s_channel_pre = 'chat_pre'
    else:
        raise NotImplementedError(s_channel)
    if t_channel == 'gen':
        t_channel_now = 'gen_1'
        t_channel_pre = 'gen_0'
    elif t_channel == 'chat':
        t_channel_now = 'chat'
        t_channel_pre = 'chat_pre'
    else:
        raise NotImplementedError(t_channel)
    print(s_uid,s_channel,t_uid,t_channel,extra_params)
    copy_state(s_uid,s_channel_now,t_uid,t_channel_now)
    copy_state(s_uid,s_channel_pre,t_uid,t_channel_pre)
    params = load_params(s_uid,s_channel)
    params = copy.deepcopy(params)
    fix_params_mismatch(params,t_channel,extra_params)
    save_params(t_uid,t_channel,**params)

def save_archive_state_and_params(uid,channel,arch_name,arch_uid=None,max_archive=MAX_USER_ARCHIVE):
    """
    Archive state and params of [{uid}_{channel}] into [arch_{arch_uid}][arch_name].
    If arch_uid=None, the arch_uid will be set to uid.
    """
    state_n = f'{uid}_{channel}'
    if channel == 'gen':
        state_n = f'{uid}_gen_1'
        state_pre_n = f'{uid}_gen_0'
    elif channel == 'chat':
        state_n = f'{uid}_chat'
        state_pre_n = f'{uid}_chat_pre'
    else:
        raise NotImplementedError(channel)
    params_n = f'params_{uid}_{channel}'
    state = copy.deepcopy(all_state[state_n])
    state_pre = copy.deepcopy(all_state[state_pre_n])
    params = copy.deepcopy(all_state[params_n])
    arch_uid = arch_uid or uid
    arch_n = f"arch_{arch_uid}"
    if arch_n not in all_state:
        all_state[arch_n] = {}
    if max_archive is None or len(all_state[arch_n])<max_archive:
        all_state[arch_n][arch_name]={"state":state,"state_pre":state_pre,"params":params,"time":time.strftime('%Y%m%d %H:%M:%S')}
        dump_all_state()
    else:
        raise Exception(f"Archives exceed {max_archive}.")
    
def del_archive_state_and_params(arch_uid,arch_name):
    arch_n = f'arch_{arch_uid}'
    if arch_n in all_state and arch_name in all_state[arch_n]:
        del all_state[arch_n][arch_name]
        dump_all_state()
    else:
        raise LookupError(arch_n,arch_name)

def load_archive_state_and_params(uid,channel,arch_name,arch_uid=None,extra_params=''):
    state_n = f'{uid}_{channel}'
    if channel == 'gen':
        state_n = f'{uid}_gen_1'
        state_pre_n = f'{uid}_gen_0'
    elif channel == 'chat':
        state_n = f'{uid}_chat'
        state_pre_n = f'{uid}_chat_pre'
    else:
        raise NotImplementedError
    params_n = f'params_{uid}_{channel}'
    arch_uid = arch_uid or uid
    arch_n = f"arch_{arch_uid}"
    if arch_n in all_state and arch_name in all_state[arch_n]:
        all_state[state_n] = copy.deepcopy(all_state[arch_n][arch_name]['state'])
        all_state[state_pre_n] = copy.deepcopy(all_state[arch_n][arch_name]['state_pre'])
        params = copy.deepcopy(all_state[arch_n][arch_name]['params'])
        fix_params_mismatch(params,channel,extra_params)
        all_state[params_n] = params
    else:
        raise LookupError(arch_n,arch_name)

def copy_archive_state_and_params(s_arch_uid,s_arch_name,t_arch_uid,t_arch_name,max_archive=MAX_USER_ARCHIVE):
    s_arch_n = f"arch_{s_arch_uid}"
    t_arch_n = f"arch_{t_arch_uid}"
    if s_arch_n in all_state and s_arch_name in all_state[s_arch_n]:
        if t_arch_n not in all_state:
            all_state[t_arch_n] = {}
        if max_archive is None or len(all_state[t_arch_n])<max_archive:
            all_state[t_arch_n][t_arch_name] = all_state[s_arch_n][s_arch_name]
        else:
            raise Exception(f"Archives exceed {max_archive}.")
    else:
        raise LookupError(s_arch_uid,s_arch_name)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


#def fix_tokens(tokens):
#    if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
def fix_tokens_end_line(tokens):
    if not tokenizer.is_trie() and tokens and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
        # print("Tokens fixed")
    return tokens

def fix_tokens_end_text(tokens):
    if tokens and tokens[-1] == END_OF_TEXT:
        if tokenizer.is_trie():
            tokens = tokens[:-1] + [END_OF_LINE_DOUBLE_TRIE]
        else:
            tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens


precomputed_count = 0
def check_precomputed_size():
    global precomputed_count
    if args.MAX_PRECOMPUTED_STATE and precomputed_count>args.MAX_PRECOMPUTED_STATE:
        for key in [key for key in all_state if key.startswith('_')]:
            print(f"- Removing precomputed state {key} due to MAX_PRECOMPUTED_STATE")
            del all_state[key]
            precomputed_count-=1
            if precomputed_count<args.MAX_PRECOMPUTED_STATE:
                break

def ensure_scenario_state(scenario):
    try:
        out, state, tokens = load_all_state("",scenario.intro.__name__)
    except:
        check_precomputed_size()
        tokens = tokenizer.encode(scenario.intro())
        #tokens = fix_tokens(tokens)
        tokens = fix_tokens_end_line(tokens)
        out, state = run_rnn(tokens)
        save_all_state("", scenario.intro.__name__, out, state, tokens)
    return out, state, tokens


def init_run():
    try:
        recover_all_state(remove_precomputed_state=args.INIT_REMOVE_PRECOMPUTED_STATE)
        print("Recovered state")
    except Exception as e:
        print("Failed to recover state.",e)
    print("Loading chat intro...")
    ensure_scenario_state(SCENARIO_CHAT)

    print("Loading assistant intro...")
    ensure_scenario_state(SCENARIO_ASSISTANT)

    clear_cache()
    dump_all_state()


def recover_all_state(remove_precomputed_state=True):
    global all_state
    with open(args.STATE_DUMP_NAME, 'rb') as file:
        all_state = pickle.load(file)
    if remove_precomputed_state:
        print("Removing previously precomputed states")
        for key in [key for key in all_state if key.startswith('_')]:
            print(f"- Removing {key}")
            del all_state[key]


def dump_all_state():
    with open(args.STATE_DUMP_NAME, 'wb') as file:
        pickle.dump(all_state, file, protocol=pickle.HIGHEST_PROTOCOL)


def clamp(n, minimum, maximum):
    return max(minimum, min(n, maximum))


def translate_message(message, from_lang, to_lang):
    translator = translate.Translator(to_lang, from_lang)
    translated = translator.translate(message)
    if from_lang == "autodetect":
        translated = message if translated == SAME_LANG else translated
    elif from_lang != to_lang:
        print(f"translated from {from_lang}: {translated}")
    return translated


def on_reset(user: User, message: str, scenario: Scenario, sampler: SAMPLER) -> str:
    # print(scenario.user_name,scenario.bot_name,scenario.intro.__name__)
    out, model_state, model_tokens = ensure_scenario_state(scenario)
    scenario = copy.deepcopy(scenario)
    sampler = copy.deepcopy(sampler)
    message = sampler.parse(message)
    message,_ = scenario.parse(message)

    save_all_state(user.id, "chat", out, model_state, model_tokens)
    save_params(user.id, "chat", scenario=scenario, sampler=sampler)

    return f"Chat reset for {user.nickname}. You are {scenario.user_name} and I am {scenario.bot_name}."


def on_show_params(user: User, message: str) -> str:
    try:
        params = load_params(user.id, "chat")
        scenario: Scenario = params['scenario']
        sampler: SAMPLER = params['sampler']
        message = sampler.parse(message)
        message,_ = scenario.parse(message)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)
    except:
        sampler = copy.deepcopy(CHAT_SAMPLER)
        scenario = copy.deepcopy(SCENARIO_CHAT)
        message = sampler.parse(message)
        message,_ = scenario.parse(message)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)
    return str(sampler)+"\n"+str(scenario)


def on_translate(user: User, message: str) -> str:
    lang_match = re.search("\-([a-z]{2}(-[A-Z]{2})?)\s+", message)
    to_lang = "zh"

    if lang_match is not None:
        message = message.replace(lang_match.group(0), "")
        to_lang = lang_match.group(1)

    from_lang = langid.classify(message)[0]
    reply = translate_message(message, from_lang, to_lang)
    reply = f"Translated from {from_lang} to {to_lang}:\n{reply}"
    return reply


def on_generate(user: User, message: str, mode=GenerateMode.GENERATE) -> str:
    message = message.replace("\r\n", '\n').replace('\\n', '\n').strip()
    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    print(f"{user.nickname}({user.id}): {message}")

    reply: str = ""

    if mode not in [GenerateMode.RETRY, GenerateMode.MORE]:
        if mode == GenerateMode.GENERATE:
            sampler = copy.deepcopy(CHAT_SAMPLER)
        elif mode == GenerateMode.INSTRUCT:
            sampler = copy.deepcopy(INSTRUCT_SAMPLER)

        message = sampler.parse(message)
        active_mode = mode
        save_params(user.id, "gen", mode=mode, sampler=sampler)
    else:
        try:
            params = load_params(user.id, "gen")
            sampler: SAMPLER = params['sampler']
            active_mode = params['mode']

            message = sampler.parse(message)
            save_params(user.id, "gen", mode=active_mode, sampler=sampler)
        except Exception as e:
            print(e)
            return reply

    print(str(sampler))

    if mode == GenerateMode.RETRY:
        try:
            out, model_state, model_tokens = load_all_state(user.id, "gen_0")
        except:
            return reply
    elif mode == GenerateMode.MORE:
        try:
            out, model_state, model_tokens = load_all_state(user.id, "gen_1")
            save_all_state(user.id, "gen_0", out, model_state, model_tokens)
        except:
            return reply
    elif mode == GenerateMode.INSTRUCT:
        message = prompt.instruct_format(message)
        model_tokens = tokenizer.encode(message)
        out, model_state = run_rnn(model_tokens)
        save_all_state(user.id, "gen_0", out, model_state, model_tokens)
    else:
        message = '\n' + message.strip()
        model_tokens = tokenizer.encode(message)
        out, model_state = run_rnn(model_tokens)
        save_all_state(user.id, "gen_0", out, model_state, model_tokens)

    occurrence = {}
    start_time = time.time()

    begin = len(model_tokens)
    end = begin
    for i in range(MAX_GENERATE_LEN):
        if active_mode == GenerateMode.GENERATE:
            out[0] = DONT_OUTPUT
        for n in occurrence:
            out[n] -= sampler.presence_penalty + \
                occurrence[n] * sampler.count_penalty

        token = sampler.sample(out)
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        if i > sampler.penalty_range:
            return_token = model_tokens[-sampler.penalty_range]
            if return_token in occurrence:
                occurrence[return_token] -= 1
                if occurrence[return_token] == 0:
                    del occurrence[return_token]

        #model_tokens += [token]
        if token != END_OF_TEXT:
            model_tokens += [token]
        out, model_state = run_rnn([token], model_state)

        xxx = tokenizer.decode(model_tokens[end:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            end = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        #if token == 0:
        if token == END_OF_TEXT:
            break

    end_time = time.time()
    delta_time = end_time - start_time
    print(f"\nTokens: {end - begin}\nTime: {delta_time}")

    clear_cache()
    save_all_state(user.id, "gen_1", out, model_state, model_tokens)

    reply = reply.strip()
    return reply




def on_message(user: User, message: str, alt=False, prevent_derail=True) -> str:
    message = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    # message = re.sub("\n(\s*\n)+", '\n', message)

    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    if not alt and len(message) == 0:
        return ""
    print(f"{user.nickname}({user.id}): {message}")

    # lang = langid.classify(message)[0]
    reply: str = ""

    try:
        channel = "chat_pre" if alt else "chat"
        out, model_state, model_tokens = load_all_state(user.id, channel)
        
        params = load_params(user.id, "chat")
        scenario: Scenario = params['scenario']
        sampler: SAMPLER = params['sampler']
        message = sampler.parse(message)
        message,chat_format_args = scenario.parse(message)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)
    except:
        if alt:
            return reply

        scenario = copy.deepcopy(SCENARIO_CHAT)
        sampler = copy.deepcopy(CHAT_SAMPLER)
        message = sampler.parse(message)
        message,chat_format_args = scenario.parse(message)

        out, model_state, model_tokens = ensure_scenario_state(scenario)

        save_all_state(user.id, "chat", out, model_state, model_tokens)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)

    print(str(sampler))
    # print(f"{scenario.bot_name}{scenario.interface}", end='')

    if not alt:
        message = scenario.chat_format(message,**chat_format_args)
        print(message,end='')
        tokens = tokenizer.encode(message)

        model_tokens += tokens
        if tokens:#如果是空message（例如直接让模型补全）那么就不用run_run了
            out, model_state = run_rnn(tokens, model_state)

        save_all_state(
            user.id,
            "chat_pre",
            out,
            model_state,
            model_tokens)
        
    occurrence = {}
    begin = len(model_tokens)
    end = begin
    for i in range(MAX_REPLY_LEN):
        if i <= 0:
            nl_bias = DONT_OUTPUT
        elif i <= 30:
            nl_bias = (i - 30) * 0.1
        else:
            nl_bias = 0
        # else:
        #     nl_bias = (i - 300) * 0.25
        out[END_OF_LINE] += nl_bias
        for n in occurrence:
            out[n] -= sampler.presence_penalty + \
                occurrence[n] * sampler.count_penalty

        token = sampler.sample(out)
        if token != END_OF_LINE:
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

        if i > sampler.penalty_range:
            return_token = model_tokens[-sampler.penalty_range]
            if return_token in occurrence:
                occurrence[return_token] -= 1
                if occurrence[return_token] == 0:
                    del occurrence[return_token]

        #tokens = [END_OF_LINE, END_OF_LINE] if token == END_OF_TEXT else [token]
        tokens = fix_tokens_end_text([token])
        model_tokens += tokens
        out, model_state = run_rnn(tokens, model_state)

        xxx = tokenizer.decode(model_tokens[end:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            end = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if '\n\n' in reply:
            break
        if prevent_derail:
            # State recovery
            def recover_state(forbidden: str, reply: str, out, model_state, model_tokens):
                idx = reply.find(forbidden)
                if idx < 0:
                    return idx, reply, out, model_state, model_tokens

                reply = f" {reply[:idx].strip()}\n\n"
                tokens = tokenizer.encode(reply)
                #tokens = fix_tokens(tokens)
                tokens = fix_tokens_end_line(tokens)
                out, model_state, model_tokens = \
                    load_all_state(user.id, "chat_pre")

                model_tokens += tokens
                out, model_state = run_rnn(tokens, model_state)

                return idx, reply, out, model_state, model_tokens
            #check generated {user_name}: 
            idx, reply, out, model_state, model_tokens = recover_state(
                f"{scenario.user_name}{scenario.interface}",
                reply,
                out,
                model_state,
                model_tokens)
            if idx >= 0:
                print(f"\nRecovered: {tokenizer.decode(model_tokens[begin:])}")
                break
            #check generated {bot_name}:
            idx, reply, out, model_state, model_tokens = recover_state(
                f"{scenario.bot_name}{scenario.interface}",
                reply,
                out,
                model_state,
                model_tokens)
            if idx >= 0:
                print(f"\nRecovered: {tokenizer.decode(model_tokens[begin:])}")
                break

    clear_cache()
    save_all_state(user.id, "chat", out, model_state, model_tokens)

    reply = reply.replace(scenario.avatar_name, user.nickname)
    reply = reply.replace(scenario.avatar_name.lower(), user.nickname)
    reply = reply.replace(scenario.avatar_name.upper(), user.nickname)
    reply = reply.strip()
    # reply = translate_message(reply, "en", lang)
    return reply

def on_copy(s_uid, s_channel, t_uid, t_channel,extra_params=''):
    try:
        copy_all(s_uid, s_channel, t_uid, t_channel,extra_params)
        if s_uid == t_uid:
            return f"Change {s_uid}'s {s_channel} to {t_channel}. Extra parameters {extra_params}"
        else:
            return f"Copy from {s_uid}-{s_channel} to {t_uid}-{t_channel}. Extra parameters {extra_params}"
    except Exception as e:
        print(e)
        return f"Failed to copy from {s_uid}-{s_channel} to {t_uid}-{t_channel}. Extra parameters {extra_params}"
    
def on_save_archive(uid,channel,arch_name,arch_uid=None,max_archive=MAX_USER_ARCHIVE):
    try:
        save_archive_state_and_params(uid,channel,arch_name,arch_uid,max_archive)
        reply = f"Save archive for {uid}-{channel} into {arch_uid}:{arch_name}"
    except Exception as e:
        print(e)
        reply = f"Failed to save archive for {uid}-{channel} into {arch_uid}:{arch_name}. Max archive nums: {max_archive}"
    return reply

def on_del_archive(arch_uid,arch_name):
    try:
        del_archive_state_and_params(arch_uid,arch_name)
        reply = f"Delete archive {arch_uid}:{arch_name}"
    except Exception as e:
        print(e)
        reply = f"Failed to delete archive {arch_uid}:{arch_name}."
    return reply

def on_load_archive(uid,channel,arch_name,arch_uid=None,extra_params=""):
    try:
        load_archive_state_and_params(uid,channel,arch_name,arch_uid,extra_params)
        reply = f"Load archive from {arch_uid}:{arch_name} to {uid}-{channel}"
    except Exception as e:
        print(e)
        reply = f"Failed to load archive from {arch_uid}:{arch_name} to {uid}-{channel}. Extra parameters {extra_params}"
    return reply

def on_copy_archive(s_arch_uid,s_arch_name,t_arch_uid,t_arch_name,max_archive=MAX_USER_ARCHIVE):
    try:
        copy_archive_state_and_params(s_arch_uid,s_arch_name,t_arch_uid,t_arch_name,max_archive)
        reply = f"Copy archive from {s_arch_uid}:{s_arch_name} to {t_arch_uid}:{t_arch_name}"
    except Exception as e:
        print(e)
        reply = f"Failed to copy archive from {s_arch_uid}:{s_arch_name} to {t_arch_uid}:{t_arch_name}. Max archive nums: {max_archive}"
    return reply

def on_list_archive(arch_uid):
    arch_n = f"arch_{arch_uid}"
    reply = f"Archives for {arch_uid}:\n"
    if arch_n in all_state:
        reply += '\n'.join(f"{i+1}. {k} ({v['time']})" for i,(k,v) in enumerate(all_state[arch_n].items()))
    return reply

    

if __name__ == "__main__":
    init_run()
