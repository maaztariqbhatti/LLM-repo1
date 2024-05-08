from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

device = "cuda" # the device to load the model onto

model_m7b = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto",trust_remote_code = True).to("cuda")
tokenizer_m7b = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# messages = [
#     {"role": "user", "content": ""}
# ]

# encodeds = tokenizer_m7b.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# model_m7b.to(device)

# generated_ids = model_m7b.generate(model_inputs, max_new_tokens=1000, do_sample=False)
# decoded = tokenizer_m7b.batch_decode(generated_ids)

# print(decoded[0])

model_m7b.generation_config.temperature=None
model_m7b.generation_config.top_p=None

pipeline_m7b = transformers.pipeline(
        task = "text-generation",
        model = model_m7b,
        return_full_text = False,
        tokenizer = tokenizer_m7b,
        max_new_tokens = 512,
        do_sample = False
    )


prompt ="""
<s>[INST] Act as a location extractor and extract all relevant locations with respect to the user question.

Answer using following context only: 
Flood warning update Red flood warnings have now been extended to include Perthshire as well as Aberdeenshire and Angus. Amber and yellow warnings are also in place across Scotland and the National Park. For more flood updates visit #StormBabet

**FLOOD WARNING** at North Sea at North and South Shields (North Tyneside, South Tyneside)

**FLOOD WARNING** at River Maun at Haughton, Milton and West Drayton (Nottinghamshire)

Breaking: Flood warning issued along the River Don in Aberdeenshire

Angus Glens region in Scotland. The red warning hasn’t come into force yet and already we’re seeing scenes like this. Potential for historic levels of flooding. Awful.

**FLOOD WARNING** at Inverurie

Evacuations have just been ordered for the town of Brechin, Angus, alongside a severe flood warning . #Brechin #Angus #Scotlandweather #floodrisk #flooding

**FLOOD WARNING** at Blairgowrie to the River Isla

Over 300 homes in #Brechin, #Scotland, are being evacuated due to a rare red weather warning caused by #StormBabet. The region anticipates severe #flooding and life-threatening conditions. Red warnings are issued only when there is a high likelihood of dangerous weather that…

The Met Office has issued red warnings for parts of the UK for severe weather due to #StormBabet Read our top tips to help protect your car from flood damage And here's what to do if your home's been flooded Stay safe!

flooding in Forfar Angus getting pretty bad with red warning due to storm babet

Storm Babet: 350 homes in Brechin, Scotland told to evacuate after warning of extensive flooding and risk to life from Storm Babet.

Beginning to see reports of roads around Glenrothes being flooded and at least one blocked by a fallen tree. We may be just outside the full Red Warning area but an Amber Warning is still serious. Please think very carefully about whether your journey is necessary.

Possible evacuations as red warning extended Residents of Brechin may need to be evacuated if the Angus town's flood defences are at risk of being breached, Scotland's Environmental Protection Agency says. #StormBabet

Storm Babet triggers a red alert in the UK, with exceptional rainfall and flooding posing a “danger to life"

Storm Babet triggers a red alert in the UK, with exceptional rainfall and flooding posing a “danger to life"

Storm Babet - WARNINGS UPDATE The Met Office have extended the area of the RED warning for HEAVY RAIN in eastern Scotland. An AMBER warning for HEAVY RAIN has also been issued for parts of N England, the Midlands and Borders All the warning details:

Parts of the UK are being warned of 'danger to life' due to severe flooding.

Warnings increased as Storm Babet batters England Homes and businesses are likely to be flooded, and deep floodwaters could cause danger to life. #... #Babet #batters #England #increased #Storm #warnings

Ireland and UK may have months worth of rain news has said from Storm babet few areas red level warnings of flooding

A Severe Flood Warning has been issued for Brechin. Follow for travel advice Follow for local advice Visit for more information on the Severe Flood Warning and the potential impacts.

Breaking: All residents have been told to leave the town of Brechin in Angus, Scotland. A severe flood warning is set to be put in place for the Brechin River and South Esk area, Angus Council has said. Those in the affected areas should leave their homes

Mandatory evacuations have just been issued for the town of Brechin, Angus, alongside a severe flood warning . #Brechin #Angus #Scotlandweather #floodrisk #flooding

U.K. Hundreds of residents in flood risk areas of Angus are to be evacuated as Storm Babet sweeps across Scotland. A severe flood warning is set to be put in place for the River South Esk area, Angus Council has said.

The Red warning's been extended to: Aberdeenshire Angus Dundee Perth &amp; Kinross This means extremely dangerous travel conditions and floodwater could cause a danger to life. Please avoid travel in these areas. More safety advice here:

The Met Office is warning parts of the UK could be cut off by flooding as Storm Babet batters the country. A new amber warning for rain has been issued for parts of northern England, the Midlands and Wales, and a rare red weather warning is in place in Scotland.

Storm Babet: Environment Agency issues flood warning for Yorkshire as Storm Babet set to hit York, Leeds, Bradford, Halifax, Huddersfield, Sheffield and Middlesbrough THE YORKSHIRE POST - NEWS-

**FLOOD WARNING** at North Sea at Sandsend (North Yorkshire)

Localised Flood Warning issued for Findhorn, Nairn, Moray and Speyside region

Update: Five places in England have been given flood warnings: Barbourne in Worcester; the River Maun near Retford; the coastlines at Bridlington and Scarborough in Yorkshire; and the banks of the Tyne at North and South Shields in Newcastle.

Question: Which locations are receiving flood warnings?
[/INST]"""

sequence = pipeline_m7b(
    prompt
)

print(sequence[0]["generated_text"])