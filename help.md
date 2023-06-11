# Help
Model: <model>

### Misc Utilities
1. `-tr [-en, -zh] <text>`: Translate
2. `-p, -params`: Show chat parameters

### Free Generation
1. `-h, -help`: Show this help
2. `-g, -gen <text>`: Generate text
3. `-e, -retry`: Retry last generation
4. `-m, -more`: Generate more
6. `-i, -inst <text>`: Follow instructions

### Chat
1. `-s, -reset`: Reset your chat chain (Casual)
2. `-b, -bot`: Reset your chat chain (AI Assistant)
3. `-alt`: Alternative reply
4. <chat>: Chat with me

### Parameters
| Param             | Description                        | Default (Chat Mode) | Default (Bot/Instruction Mode) |
| ----------------- | ---------------------------------- | ------------------- | ------------------------------ |
| `-nucleus`        | Switch to nucleus sampling         | <chat_nucleus>      | <inst_nucleus>                 |
| `-typical`        | Switch to typical sampling         | <chat_typical>      | <inst_typical>                 |
| `-temp=<number>`  | Higher temperature → more verbose  | <chat_temp>         | <inst_temp>                    |
| `-top_p=<number>` | Lower top p → more accurate answer | <chat_top_p>        | <inst_top_p>                   |
| `-tau=<number>`   | Lower tau → more human-like answer | <chat_tau>          | <inst_tau>                     |
| `-af=<number>`    | Count penalty, avoids repeating    | <chat_af>           | <inst_af>                      |
| `-ap=<number>`    | Presence penalty                   | <chat_ap>           | <inst_ap>                      |
| `-temp_a=<number>`| Temperature (prior to top_p/tau)   | <chat_temp_a>       | <inst_temp_a>                  |

### Examples
#### Chat
* Use `-alt` to retry
* Use `-reset` or `-s` to reset, better at chillin'
* Use `-bot` or `-b` to enter bot mode, better at QA and coding
* Do not use `-retry` or `-more`: useless

```
<user>: What's your interesting news today?

<bot>: ...

<user>: -alt

<bot>: ...

<user>: -reset

<bot>： Chat reset for ...
```

#### Instruct
* Use `-retry` or `-e` to retry
* Use `-more` to continue

```
<user>: -i Write a letter to our customers, apologizing for delay in delivery of our products.

<bot>: Dear Valued Customers:
...
Yours Sincerely
[YOUR NAME]

<user>: -retry

<bot>: ...
```

#### Translation
```
<user>: -tr Hello World!

<bot>: 你好，世界！

<user>: -tr -en 你好，世界！

<bot>: Hello, world!
```

#### Generation
```
<user>: -gen -temp=2.0 -top_p=0.5 Here is a story about a man's lonely journey on Mars:

<bot>: (Some story...)

<user>: -m

<bot>: (Continue generates...)
```
    
### ChatGal激进预览版新增指令（随时可能变动）
    
#### 存档管理
    
1. `/save [chat, gen] archive_name`: Save current states into the archive. (Currently max archives per user: 10)
2. `/list`: List the archives.
3. `/load [chat, gen] archive_name`: Load current states from the archive.
4. `/del archive_name`: Delete the archive.
5. `/copy user_id channel (target_channel)`: Copy others' states to yours. `target_channel` is optional.
6. `/changefrom [chat,gen] [chat,gen]`: Change current chat to gen mode or vice versa. If change gen to chat, extra arguments `-u <user_name> -b <bot_name>` are requierd.
6. `/list_pub`: List the public archives.
7. `/load_pub [chat, gen] archive_name`: Load current states from the public archive.

#### 重置到不同的对话设定
在`/s`(重置聊天)指令下加入了一些新的子命令
    
1. `/s <预设名称>`可以切换到某个预设的对话设定。例如`/s 巧克力`切换到巧克力的默认聊天设定。可以通过`/scenarios`列出所有的预设对话设定。
2. `/s -t <模板名称> -u <用户角色> -b <机器人角色>`可以使用某些对话提示模板，自定义对话中使用的角色名称。例如/s -t catgirl_phd_en -u Kashou -b Cinnamon 将会使用”伯克利猫娘博士“的对话模板，用户使用的角色为Kashou，机器人使用的角色是Cinnamon。可以通过`/templates`列出现有所有的对话提示模板。
3. 有一个特殊的模板custom_intro，该模板可以用来添加自定义的对话前缀提示。用法示例：
```
/s -t custom_intro -u Kashou -b Chocola -i 
{Kashou made Chocola cry.}

Kashou: Why are you crying.

Chocola: Because Kashou-chan is a badass!!!
```
将上面的命令输入，则会使用-i之后的全文作为对话的前缀提示。

#### 聊天指令的增强
    
在`/c`指令下加入了一些增强功能
    
1. 可以切换用户角色和机器人角色。用法`/c -u <用户角色> -b <机器人角色> <聊天内容>`。例如`/c -b 香草`将让机器人以香草的身份回复。**注意**，现在`-u`和`-b`切换的角色会永久生效。`/cu`是`/c -u`的等效缩写，`/cb`是`/c -b`的等效缩写。
2. 可以临时输入旁白。用法`/c -n <旁白内容>`。目前旁白使用`{}`包裹，需要手动输入。例如`/c -n {我摸了摸香草的脑袋}`。 `/cn`是`/c -n`的等效缩写。
3. 可以让机器人继续给出一轮聊天回复。用法`/c -m`。例如在和巧克力的聊天中，直接输入`/c -m`可以让巧克力继续说一段话。此外该命令可以配合`-b`和`-n`指令。`/c -m -b <机器人角色>`可以让机器人切换到指定角色回复一段话。`/c -m -n`会让模型自动补全一段话。`/cm` `/cmb` `/cmn`是以上命令的等效缩写。（目前暂时没有让机器人直接生成旁白的办法，但是可以使用`/cmn {`的方式让模型尝试补全出一段旁白。）
4. 可以切换用户的化身角色，用法`/c -n <化身角色名>`。聊天中出现的化身角色名称将被替换成用户的昵称。例如黄鹤通过`/c -a 香草`后，之后机器人回复中提到的`香草`都会被替换成`黄鹤`。
