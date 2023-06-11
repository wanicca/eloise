import os
import requests
import chat
import re
import datetime
import logging
import traceback

from prompt import User, SCENARIO_CHAT, SCENARIO_ASSISTANT, get_scenario,PREDEFINED_SCENARIOS_MAIN_NAMES, PREDEFINED_INTRO_TEMPLATES
from chat import GenerateMode, CHAT_SAMPLER, INSTRUCT_SAMPLER, model

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.FileHandler(f"logs/eloise-{datetime.date.today()}.txt")
handler.setFormatter(formatter)

logger = logging.getLogger("eloise")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


banned_users = []
banned_groups = []
non_chat_groups = []


CHAT_HELP_COMMAND = "`/c, /chat <text>`"
PRIVATE_HELP_COMMAND = "`<text>`"

with open("./help.md", 'r') as file:
    model_name = model.args.MODEL_NAME.split('/')[-1].replace('.pth', '')

    HELP_MESSAGE = file.read()
    HELP_MESSAGE = HELP_MESSAGE.replace('<model>', model_name)
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_nucleus>', 'Yes' if CHAT_SAMPLER.sample.__name__ == "sample_nucleus" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_typical>', 'Yes' if CHAT_SAMPLER.sample.__name__ == "sample_typical" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_temp>', str(CHAT_SAMPLER.temp))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_top_p>', str(CHAT_SAMPLER.top_p))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_tau>', str(CHAT_SAMPLER.tau))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_af>', str(CHAT_SAMPLER.count_penalty))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_ap>', str(CHAT_SAMPLER.presence_penalty))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_nucleus>', 'Yes' if INSTRUCT_SAMPLER.sample.__name__ == "sample_nucleus" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_typical>', 'Yes' if INSTRUCT_SAMPLER.sample.__name__ == "sample_typical" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_temp>', str(INSTRUCT_SAMPLER.temp))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_top_p>', str(INSTRUCT_SAMPLER.top_p))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_tau>', str(INSTRUCT_SAMPLER.tau))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_af>', str(INSTRUCT_SAMPLER.count_penalty))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_ap>', str(INSTRUCT_SAMPLER.presence_penalty))


def commands(user: User, message, enable_chat=False, is_private=False,is_privilege=False):
    help_match = re.match("\-h(elp)?\\b", message)
    params_match = re.match("\-p(arams)?\\b", message)

    translate_match = re.match("\-tr\\b", message)
    retry_match = re.match("\-(retry|e)\\b", message)
    more_match = re.match("\-m(ore)?\\b", message)
    gen_match = re.match("\-g(en)?\s+", message)
    inst_match = re.match("\-i(nst)?\s+", message)

    # reset_match = re.match("\-(reset|s)\\b", message)
    reset_match = re.match(r"-(reset|s)(\s+((-t(emplate)?\s+(?P<intro>\S+)\s+-u(ser)?(\=|\s+)(?P<user_name>\S+)\s+-b(ot)?(\=|\s+)(?P<bot_name>\S+)(\s+-a(vatar)?(\=|\s+)(?P<avatar_name>\S+))?(?P<custom_intro>\s+-i(ntro)?(\=|\s+))?)|(?P<scenario_name>[^-\s]\S+))|\b)", message)
    reset_bot_match = re.match("\-(bot|b)\\b", message)
    alt_match = re.match("\-a(lt)?\\b", message)
    chat_match = re.match("\-c(hat)?\s+", message)
    chatalias_match = re.match("\-c(m|n|u|b|mb|mn)(\s+|$)", message)
    list_scenario_match = re.match(r"-scenarios\b",message)
    list_template_match = re.match(r"-templates\b",message)
    change_match = re.match(r"-change(?:from\s+(?P<s_channel>\S+))?\s+(?P<t_channel>\S+)",message) #Change my channel state (sampler, scenario setting)
    copy_match = re.match(r"-copy\s+(?P<s_uid>\S+)\s+(?P<s_channel>\S+)(?:\s+(?P<t_channel>\S+))?",message) #Copy current states of others to me
    save_archive_match = re.match(r"-save\s+(?P<channel>\S+)\s+(?P<arch_name>\S+)",message)#Save archive
    del_archive_match = re.match(r"-del\s+(?P<arch_name>\S+)",message)#Delete archive
    load_archive_match = re.match(r"-load\s+(?P<channel>\S+)\s+(?P<arch_name>\S+)",message)#Load archive
    list_archive_match = re.match(r"-list\b",message)#List archive
    
    load_pub_archive_match = re.match(r"-load_pub\s+(?P<channel>\S+)\s+(?P<arch_name>\S+)",message)#load public archive
    list_pub_archive_match = re.match(r"-list_pub\b",message)#list public archives
    #High Privilege!
    ##Mannage public archives
    save_pub_archive_match = re.match(r"-save_pub\s+(?P<channel>\S+)\s+(?P<arch_name>\S+)",message)
    del_pub_archive_match = re.match(r"-del_pub\s+(?P<arch_name>\S+)",message)
    ##Super commands
    supercopy_match = re.match(r"-supercopy\s+(?P<s_uid>\S+)\s+(?P<s_channel>\S+)\s+(?P<t_uid>\S+)\s+(?P<t_channel>\S+)",message) 
    supersave_archive_match = re.match(r"-supersave\s+(?P<uid>\S+)\s+(?P<channel>\S+)\s+(?P<arch_name>\S+)\s+(?P<arch_uid>\S+)",message)
    superdel_archive_match = re.match(r"-superdel\s+(?P<arch_uid>\S+)\s+(?P<arch_name>\S+)",message)
    superload_archive_match = re.match(r"-superload\s+(?P<uid>\S+)\s+(?P<channel>\S+)\s+(?P<arch_name>\S+)\s+(?P<arch_uid>\S+)",message)
    superlist_archive_match = re.match(r"-superlist\s+(?P<arch_uid>\S+)",message)
    supercopy_archive_match = re.match(r"-supercopyarch\s+(?P<s_arch_uid>\S+)\s+(?P<s_arch_name>\S+)\s+(?P<t_arch_uid>\S+)\s+(?P<t_arch_name>\S+)",message)
    dump_all_state_match = re.match(r"-dump\b",message)
    
    
    help = HELP_MESSAGE
    if is_private:
        help = help.replace('<chat>', PRIVATE_HELP_COMMAND)
    else:
        help = help.replace('<chat>', CHAT_HELP_COMMAND)

    prompt = message
    reply = ""
    matched = True

    if help_match:
        reply = help
    elif params_match:
        prompt = message[params_match.end():]
        reply = chat.on_show_params(user, prompt)
    elif translate_match:
        prompt = message[translate_match.end():]
        reply = chat.on_translate(user, prompt)
    elif retry_match:
        prompt = message[retry_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.RETRY)
    elif more_match:
        prompt = message[more_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.MORE)
    elif gen_match:
        prompt = message[gen_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.GENERATE)
    elif enable_chat and inst_match:
        prompt = message[inst_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.INSTRUCT)
    elif enable_chat and reset_bot_match:
        prompt = message[reset_bot_match.end():]
        reply = chat.on_reset(user, prompt, SCENARIO_ASSISTANT, CHAT_SAMPLER)
    elif enable_chat and reset_match:
        groupdict = reset_match.groupdict()
        prompt = message[reset_match.end():]
        if groupdict['custom_intro']:
            groupdict['custom_intro'] = prompt
            prompt = ''
        try:
            scenario = get_scenario(**groupdict)
        except Exception as e:
            scenario = SCENARIO_CHAT
        reply = f"Scenario: {scenario.intro.__name__}\n\n"
        reply += chat.on_reset(user, prompt, scenario, CHAT_SAMPLER)
    elif enable_chat and alt_match:
        prompt = message[alt_match.end():]
        reply = chat.on_message(user, prompt, alt=True)
    elif enable_chat and is_private:
        reply = chat.on_message(user, prompt)
    elif enable_chat and not is_private and chat_match:
        prompt = message[chat_match.end():]
        reply = chat.on_message(user, prompt)
    elif enable_chat and not is_private and chatalias_match:
        prompt = ""
        for c in chatalias_match.group(1):
            prompt += f"-{c} "        
        prompt += message[chatalias_match.end():]
        reply = chat.on_message(user, prompt, prevent_derail=False)
    elif list_scenario_match:
        reply = "Predefined scenarios:\n\n" + "\n".join(PREDEFINED_SCENARIOS_MAIN_NAMES)
    elif list_template_match:
        reply = "Predefined prompt template:\n\n" + "\n".join(PREDEFINED_INTRO_TEMPLATES.keys())
    elif change_match:
        groupdict = change_match.groupdict()
        groupdict['s_uid'] = user.id
        groupdict['t_uid'] = user.id
        groupdict['s_channel'] = groupdict.get('s_channel') or groupdict['t_channel']
        groupdict['extra_params'] = message[change_match.end():].strip()
        reply = chat.on_copy(**groupdict)
    elif copy_match or supercopy_match:
        if copy_match:
            groupdict = copy_match.groupdict()
            groupdict['t_uid'] = user.id
            if groupdict.get('t_channel') is None:
                groupdict['t_channel'] = groupdict['s_channel']
            groupdict['extra_params'] = message[copy_match.end():].strip()
        elif supercopy_match and is_privilege :
            groupdict = supercopy_match.groupdict()
            groupdict['extra_params'] = message[supercopy_match.end():].strip()
        reply = chat.on_copy(**groupdict)
    elif save_archive_match or save_pub_archive_match or supersave_archive_match:
        if save_archive_match:
            groupdict = save_archive_match.groupdict()
            groupdict['uid'] = user.id
            groupdict['arch_uid']=user.id
        elif save_pub_archive_match and is_privilege:
            groupdict = save_pub_archive_match.groupdict()
            groupdict['uid'] = user.id
            groupdict['arch_uid']='pub'
            groupdict['max_archive'] = None
        elif supersave_archive_match and is_privilege:
            groupdict = supersave_archive_match.groupdict()
            groupdict['max_archive'] = None
        reply = chat.on_save_archive(**groupdict)
    elif del_archive_match or del_pub_archive_match or superdel_archive_match:
        if del_archive_match:
            groupdict = del_archive_match.groupdict()
            groupdict['arch_uid'] = user.id
        elif del_pub_archive_match and is_privilege:
            groupdict = del_pub_archive_match.groupdict()
            groupdict['arch_uid'] = 'pub'
        elif superdel_archive_match and is_privilege:
            groupdict = superdel_archive_match.groupdict()
        reply = chat.on_del_archive(**groupdict)
    elif load_archive_match or load_pub_archive_match or superload_archive_match:
        if load_archive_match:
            groupdict = load_archive_match.groupdict()
            groupdict['uid'] = user.id
            groupdict['arch_uid']=user.id
            groupdict['extra_params'] = message[load_archive_match.end():].strip()
        elif load_pub_archive_match:
            groupdict = load_pub_archive_match.groupdict()
            groupdict['uid'] = user.id
            groupdict['arch_uid'] = 'pub'
            groupdict['extra_params'] = message[load_pub_archive_match.end():].strip()
        elif superload_archive_match  and is_privilege:
            groupdict = superload_archive_match.groupdict()
            groupdict['extra_params'] = message[superload_archive_match.end():].strip()
        reply = chat.on_load_archive(**groupdict)
    elif list_archive_match or list_pub_archive_match or superlist_archive_match:
        if list_archive_match:
            groupdict = list_archive_match.groupdict()
            groupdict['arch_uid'] = user.id
        elif list_pub_archive_match:
            groupdict = list_pub_archive_match.groupdict()
            groupdict['arch_uid'] = 'pub'
        elif superlist_archive_match and is_privilege:
            groupdict = superlist_archive_match.groupdict()
        reply = chat.on_list_archive(**groupdict)
    elif supercopy_archive_match and is_privilege:
        groupdict = supercopy_archive_match.groupdict()
        groupdict['max_archive'] = None
        reply = chat.on_copy_archive(**groupdict)
    elif dump_all_state_match and is_privilege:
        reply = f"Dump all state to {chat.args.STATE_DUMP_NAME}"
        chat.dump_all_state()
    else:
        matched = False

    return matched, prompt, reply

chat.init_run()
print("Starting server...")

from simple_websocket_server import WebSocketServer, WebSocket
import json

class chatgal(WebSocket):
    def handle(self):
        try:
            req = json.loads(self.data)

            message = req['raw_message']
            sender = req['sender']

            if req['reload_lora']:
                with open("./lora_cfg.json", "w") as outfile:
                    outfile.write(json.dumps(req))
                # os.remove("./state_7b")
                quit()

            print(sender)
            print(message)

            user = User(sender)
            is_privilege = sender.get('permission') in ("OWNER","ADMINISTRATOR")

            matched, prompt, reply = commands(
                user, message, enable_chat=True, is_private=False,is_privilege=is_privilege)

            if matched:
                res = json.dumps({"response": reply},ensure_ascii=False)
                logger.info(str(sender))
                logger.info(str(message))
                logger.info(res)
                self.send_message(res)
        except Exception as e:
            traceback.print_exc()

    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')

server = WebSocketServer('', 6006, chatgal)
server.serve_forever()
