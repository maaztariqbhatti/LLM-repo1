#FSD_1777
"""
FSD-1555 
Event start: 
Event end: 

Pre flood peak (560 tweets)
Date from:
2023-06-01 06:00:00+00:00
date to
2023-06-01 13:00:00+00:00
"""

#Pre flood peak FSD_1555 -32
gt_FSD1555_peak_fw = ["Kyoto, Saitama, Oita, Uwajima, Kanagawa, Shizuoka, Yamanashi Prefecture, Tochigi Prefecture, Ibaraki Prefecture, Fukfui Prefecture, Kanto Koshin, Gunma Prefecture, Okinawa, China and Sichuan, Toyama Prefecture, Mie Aichi, Nakaechi and Uechi,Ehime Prefecture, Gifu Prefecture, Wakayama Prefecture, Southern Kyushu, Amami Prefecture, Hyogo Prefecture, Nara Prefecture, Nagano Prefecture, Shiga Prefecture, Miyazaki Prefecture, Shikoku Prefecture, Kinki Prefecture, Eastern Japan, Chiba, Nagawa River Anom City, Osaka, Nerima"]

# 23
gt_FSD1555_peak_evac = ["""Hamamatsu, Nagano, Oiso, Ueo City, Japan, Kansai, Kanto, Shizuoka, Okinawa, Tokai, Uwajima, Tokushima Prefecture, Nagaoka City and Mitsui City, China and Shikoku, Shiga,Toru City, Osaka, Chiba Prefecture, Toyama, Koga, Konan, Shinraku, Ueno, Kiso River"""]

#8
gt_FSD1555_peak_linear_rain = ["China, Shikoku, Kyushu Koshin,Nakashikoku,Ibaraki,Sichuan,Osaka,Kanto"]

#Testing evaluations array
prep_evac = """
Hamamatsu, Nagano, Oiso or Gotaku, Ueo City, Japan, Kansai, Kanto, Shizuoka, Okinawa, Tokai, Uwajima, Tokushima Prefecture, Nagaoka City and Mitsui City, China and Shikoku, Shiga,Toru City, Osaka, Chiba Prefecture, Toyama, Koga, Konan, Shinraku, Ueno, Kiso River"""

peak_fw = ["Kyoto, Saitama, Oita, Uwajima, Kanagawa, Shizuoka, Yamanashi Prefecture, Tochigi Prefecture, Ibaraki Prefecture, Fukui Prefecture, Kanto Koshin, Gunma Prefecture, Okinawa, China and Sichuan, Toyama Prefecture, Mie Aichi, Niigata (Nakaechi and Uechi),Ehime Prefecture, Gifu Prefecture, Wakayama Prefecture, Southern Kyushu, Amami Prefecture, Hyogo Prefecture, Nara Prefecture, Nagano Prefecture, Shiga Prefecture, Miyazaki Prefecture, Shikoku Prefecture, Kinki Prefecture, Eastern Japan,Chiba, Tokai, Osaka, Nerima"]


#FSD_1777
"""
FSD-1777 
Event start: 2023-10-19
Event end: 2023-10-24

Post pre flood peak (200 tweets)
Date from:
2023-10-19 18:58:40+00:00
date to
2023-10-19 23:58:47+00:00

Pre flood peak (604 tweets)
Date from:
2023-10-19 09:00:00+00:00
date to
2023-10-19 18:00:00+00:00
"""

#Pre-flood peak
#Evacuation orders or already evacuated
gt_FSD1777_casualty = ["Water of Lee, Glen Esk, Angus"]

gt_FSD1777_peak_evac = ["Brechin, Angus, Scotland, Aberdeenshire"] #Aberdeenshire already evacuated, villages of Tannadice and Finavon (told to leave no specific evacuation orders!)


gt_FSD1777_peak_fw = ["Aberdeenshire, Greater Manchester, Inverurie, River Maun at Haughton, Milton and West Drayton (Nottinghamshire), Angus, Angus Glens, Perthshire, North Sea at Sandsend (North Yorkshire),  North Tyneside, South Tyneside, Brechin, Dundee, Blairgowrie, River South Esk, Lancashire, Cheshire, York, Leeds, Bradford, Halifax, Huddersfield, Sheffield, Middlesbrough, Perth, Kinross, Barbourne in Worcester, River Maun near Retford, Bridlington and Scarborough in Yorkshire, North and South Shields in Newcastle,Findhorn, Nairn, Moray, Speyside"] # 34

#Post pre-flood peak
#Flood warning
gt_FSD1777_postpeak_fw = ["Scotland, Essex, Ireland, Angus, Dundee,UK, Perthshire, Aberdeenshire, Dundee, Stonehaven, A90, Middleton, Perth,Sheffield, Brechin, River Spen, Alyth, South Esk, North Esk, Marykirk, Logie Mill, Craigo, Finavon,Tannadice"]
#Evacuation orders or already evacuated
gt_FSD1777_postpeak_evac = ["Angus, Brechin, Dundee"]       #Dundee already evacuated 

