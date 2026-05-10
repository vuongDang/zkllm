import json                                                                 
from transformers import AutoTokenizer                                      
                                                                            
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',       
local_files_only=True)                                                      
                                                                            
base_texts = [                                                              
    """It was a dark and stormy night aboard the ISS when Commander Elena   
Vasquez noticed something impossible: the airlock indicator showed a       
suit outside, but the manifest listed every astronaut as inside. She
floated toward the porthole, heart hammering, and pressed her face to the
cold glass. There, tumbling slowly against the infinite black, was a
figure in a suit with no mission patch, no national flag, no identifying
marks whatsoever. She keyed her radio with trembling fingers. Houston,
she said, we have a situation that is not in any procedure manual I have
ever read. The response from mission control was static for eleven
unbearable seconds before a calm voice replied: We see it too, Commander.
We have been seeing it on the long-range cameras for three days and we
did not want to alarm the crew. The figure outside raised one gloved
hand. It was waving. Elena pressed her palm flat against the glass and
waved back, because what else do you do when a ghost knocks on the door
of your spaceship floating two hundred and fifty miles above the Earth?
The storm on the planet below crackled with lightning, illuminating
continents she had memorized but now felt she had never truly seen.
Whatever was outside the airlock did not seem threatening. It simply
seemed lonely, and that, somehow, was the most terrifying thing of
# all."""]
#     """The recipe had been in Grandma Rosario's family for four hundred
# years, scrawled in faded ink on a page torn from a monastery cookbook. It
# called for twelve ingredients that no longer existed by their original
# names, two techniques that required equipment discontinued in the
# nineteenth century, and one step written only as: speak kindly to the
# dough. Carlos had been trying to recreate it for seven years. He had
# consulted food historians, chemists, a retired Franciscan friar, and one
# very confused molecular gastronomist who kept insisting that bread could
# not technically have feelings. Tonight he was trying again. The kitchen
# smelled of wild yeast and rosemary and something older, something he
# could not name, like stone churches and summer thunder. He kneaded the
# dough and talked to it the way his grandmother had taught him, telling it
# about his week, his small worries, the neighbor's cat who kept stealing
# tomatoes from his garden. The dough felt different under his hands.
# Warmer. Alive in a way flour and water and time could not fully explain.
# When he pulled the loaf from the oven at midnight, the crust had formed a
# pattern he recognized from the monastery page: a sun with twelve rays,
# pressed there by no hand he could account for. He cut a slice, took a
# bite, and for one impossible moment he was four years old and Grandma
# Rosario was still alive and everything smelled like Sunday.""",
#     """Professor Amelia Chen had seventeen minutes to explain to the
# Nobel committee why her experiment had technically worked but had also,
# as a side effect, made Tuesday disappear. Not the concept of Tuesday, she
# clarified, adjusting her glasses and clicking to slide three of her
# presentation. Tuesdays continued to exist, calendars still showed them,
# people still dreaded them in the usual way. What had vanished was any
# memory, anywhere on Earth, of anything that had actually happened on a
# Tuesday. Every diary entry for a Tuesday was blank. Security footage from
# Tuesdays showed empty rooms regardless of what witnesses remembered. The
# day persisted structurally but had been emptied of content like a bottle
# poured out. She clicked to slide four. The good news, she said, is that
# we have conclusively demonstrated that temporal memory is a separable
# substrate from temporal sequence, which was the original hypothesis. The
# bad news, said the committee chair, is that your spouse's birthday was a
# Tuesday. Yes, said Professor Chen. We are working through some things.
# She clicked to slide five, which was a graph, because graphs were easier
# than eye contact. The interesting implication, she continued, her voice
# only slightly unsteady, is that if Tuesday could be emptied, the
# mechanism theoretically runs in reverse. She looked up. I believe we can
# fill it back in.""",
# ]

TARGET = 256
TARGET -= 1 
results = []
for text in base_texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:TARGET]
    if len(ids) < TARGET:
        ids += [tokenizer.eos_token_id] * (TARGET - len(ids))
    decoded = tokenizer.decode(ids)
    check = tokenizer.encode(decoded, add_special_tokens=False)
    print(f'Token count: {len(check)}')
    results.append(decoded)

with open('/home/zkllm-ccs2024/inputs.json', 'w') as f:
    json.dump(results, f, indent=2)

print('inputs.json written.')