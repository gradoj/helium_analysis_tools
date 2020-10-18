"""

Functions to print information about a specific hotspot


"""
import argparse

import utils
from math import log10, pi
from classes.Hotspots import Hotspots
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
import folium
from folium.features import DivIcon
from folium import IFrame
import os
from glob import glob
import h3
import shapely
import geojson
import base64
from shapely.geometry.point import Point
from shapely import geometry


def __heading2str__(heading):
    headingstr = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    heading = 5 * round(heading / 5, 0)
    idx = int(round(heading / 45)) % 8

    return f"{heading:3.0f} {headingstr[idx]:>2}"


def poc_summary(hotspot, chals):

    haddr = hotspot['address']
    init_target = 0
    challenger_count = 0
    last_target = None
    last_challenger = None
    max_target_delta = 0
    max_challenger_delta = 0
    untargetable_count = 0
    print(f"PoC Summary Report for: {hotspot['name']}")
    planned_count = [0] * 5
    tested_count = [0] * 5
    passed_count = [0] * 5
    for c in chals:
        if c['path'][0]['challengee'] == haddr:
            if last_target is None:
                last_target = c['height']
            else:
                max_target_delta = max(max_target_delta, last_target - c['height'])
                last_target = c['height']
            init_target += 1
        elif c['challenger'] == haddr:
            if last_challenger is None:
                last_challenger = c['height']
            else:
                challenger_delta = last_challenger - c['height']
                #TODO: reference actual chain variable not hardcoded numbers
                if challenger_delta > 300:
                    untargetable_count += challenger_delta - 300
                max_challenger_delta = max(max_challenger_delta, challenger_delta)
                last_challenger = c['height']
            challenger_count += 1

        next_passed = False
        next_addr = ''
        for i in range(len(c['path'])-1, -1, -1):

            passed = c['path'][i]['witnesses'] or (c['path'][i]['receipt'] and i > 0) or next_passed
            if c['path'][i]['challengee'] == hotspot['address']:
                planned_count[i] += 1
                if passed:
                    passed_count[i] += 1

            if passed and next_addr == hotspot['address']:
                tested_count[i + 1] += 1

            next_addr = c['path'][i]['challengee']
            next_passed = passed
        if c['path'][0]['challengee'] == hotspot['address']:
            tested_count[0] += 1

    print()
    print('PoC Eligibility:')
    tgt_percent_str = ''
    if init_target:
        tgt_percent_str = f"(every {(chals[0]['height']-chals[-1]['height'])/init_target:.0f} blocks)"
    print(f"successfully targeted   {init_target} times in {(chals[0]['height']-chals[-1]['height'])} blocks {tgt_percent_str}")
    print(f"\tlongest untargeted stretch: {max_target_delta:4d} blocks")
    chal_percent_str = ''
    if challenger_count:
        chal_percent_str = f"(every {(chals[0]['height']-chals[-1]['height'])/challenger_count:.0f} blocks)"
    print(f"challenger receipt txn  {challenger_count} times in {(chals[0]['height']-chals[-1]['height'])} blocks {chal_percent_str}")
    print(f"\tlongest stretch without challenger receipt: {max_challenger_delta:4d} blocks")
    if chals[0]['height']-chals[-1]['height']:
        print(f"\thotspot was untargetable for: {untargetable_count} blocks ({untargetable_count*100/(chals[0]['height']-chals[-1]['height']):.1f}% of blocks)")

    print()
    print(f"PoC Hop Summary:")
    print(f"Hop | planned | tested (%) | passed (%) |")
    #print(f"    |         |   \\planned |    \\tested |")
    print(f'-----------------------------------------')
    for i in range(0, 5):
        line = f"{i + 1:3} | {planned_count[i]:6d}  |"
        if not planned_count[i]:
            line += f" {tested_count[i]:3d} ( {'N/A':3}) | {passed_count[i]:3d} ( {'N/A'}) |"
        else:
            line += f" {tested_count[i]:3d} ({tested_count[i]*100/planned_count[i]:3.0f}%) |"
            if not tested_count[i]:
                line += f" {passed_count[i]:3d} ( {'N/A'}) |"
            else:
                line += f' {passed_count[i]:3d} ({passed_count[i] * 100 / tested_count[i]:3.0f}%) |'
        print(line)


def pocv10_violations(hotspot, chals):
    """

    :param hotspot: hotspot object to analyze
    :param chals: list of challenges
    :return:
    """
    H = Hotspots()
    haddr = hotspot['address']
    hlat, hlng = hotspot['lat'], hotspot['lng']
    transmits_w = dict(total=0, bad_rssi=0, bad_snr=0)
    receives_w = dict(total=0, bad_rssi=0, bad_snr=0)
    poc_rcv = dict(total=0, bad_rssi=0, bad_snr=0)
    bad_neighbors = dict()


    for chal in chals:
        transmitter = None
        for p in chal['path']:
            if p['challengee'] == haddr:
                for w in p['witnesses']:
                    dist = utils.haversine_km(
                        hlat, hlng,
                        H.get_hotspot_by_addr(w['gateway'])['lat'], H.get_hotspot_by_addr(w['gateway'])['lng'])
                    if dist < .3:
                        continue
                    rssi_lim = utils.max_rssi(dist)
                    snr_rssi_lim = utils.snr_min_rssi(w['snr'])
                    transmits_w['total'] += 1
                    if w['gateway'] not in bad_neighbors:
                        bad_neighbors[w['gateway']] = dict(rssi=0, snr=0, ttl=0)
                    bad_neighbors[w['gateway']]['ttl'] += 1
                    if w['signal'] > rssi_lim:
                        transmits_w['bad_rssi'] += 1
                        bad_neighbors[w['gateway']]['rssi'] += 1
                    if w['signal'] < snr_rssi_lim:
                        transmits_w['bad_snr'] += 1
                        bad_neighbors[w['gateway']]['snr'] += 1
                if p['receipt'] and transmitter:
                    dist = utils.haversine_km(
                        hlat, hlng,
                        H.get_hotspot_by_addr(transmitter)['lat'], H.get_hotspot_by_addr(transmitter)['lng']
                    )
                    rssi_lim = utils.max_rssi(dist)
                    snr_rssi_lim = utils.snr_min_rssi(p['receipt']['snr'])
                    poc_rcv['total'] += 1
                    if transmitter not in bad_neighbors:
                        bad_neighbors[transmitter] = dict(rssi=0, snr=0, ttl=0)
                    bad_neighbors[transmitter]['ttl'] += 1
                    if p['receipt']['signal'] > rssi_lim:
                        poc_rcv['bad_rssi'] += 1
                        bad_neighbors[transmitter]['rssi'] += 1
                    if p['receipt']['signal'] < snr_rssi_lim:
                        poc_rcv['bad_snr'] += 1
                        bad_neighbors[transmitter]['snr'] += 1
            else:
                for w in p['witnesses']:
                    if w['gateway'] != haddr:
                        continue
                    dist = utils.haversine_km(
                        hlat, hlng,
                        H.get_hotspot_by_addr(p['challengee'])['lat'], H.get_hotspot_by_addr(p['challengee'])['lng']
                    )
                    if dist < .3:
                        continue
                    rssi_lim = utils.max_rssi(dist)
                    snr_rssi_lim = utils.snr_min_rssi(w['snr'])
                    receives_w['total'] += 1
                    if p['challengee'] not in bad_neighbors:
                        bad_neighbors[p['challengee']] = dict(rssi=0, snr=0, ttl=0)
                    bad_neighbors[p['challengee']]['ttl'] += 1
                    if w['signal'] > rssi_lim:
                        receives_w['bad_rssi'] += 1
                        bad_neighbors[p['challengee']]['rssi'] += 1
                    if w['signal'] < snr_rssi_lim:
                        receives_w['bad_snr'] += 1
                        bad_neighbors[p['challengee']]['snr'] += 1
            transmitter = p['challengee']


    print(f"PoC v10 failures for {hotspot['name']}")

    print(F"SUMMARY")
    print(f"Category                   | Total | bad RSSI (%) | bad SNR (%) |")
    print(f"-----------------------------------------------------------------")
    print(f"Witnesses to hotspot >300m | {transmits_w['total']:5d} | {transmits_w['bad_rssi']:4d} ({transmits_w['bad_rssi']*100/max(1, transmits_w['total']):3.0f}%)  | {transmits_w['bad_snr']:4d} ({transmits_w['bad_snr']*100/max(1, transmits_w['total']):3.0f}%) |")
    print(f"Hotspot witnessing  >300m  | {receives_w['total']:5d} | {receives_w['bad_rssi']:4d} ({receives_w['bad_rssi']*100/max(1, receives_w['total']):3.0f}%)  | {receives_w['bad_snr']:4d} ({receives_w['bad_snr']*100/max(1, receives_w['total']):3.0f}%) |")
    print(f"Hotspot PoC receipts       | {poc_rcv['total']:5d} | {poc_rcv['bad_rssi']:4d} ({poc_rcv['bad_rssi']*100/max(1, poc_rcv['total']):3.0f}%)  | {poc_rcv['bad_snr']:4d} ({poc_rcv['bad_snr']*100/max(1, poc_rcv['total']):3.0f}%) |")

    print()
    print()
    print(f'BY "BAD" NEIGHBOR')
    print(f"Neighboring Hotspot           | owner | dist km | heading |  bad RSSI (%)  |  bad SNR (%)   |")
    print(f"------------------------------+-------+---------+---------+----------------+----------------|")
    hlat, hlng = hotspot['lat'], hotspot['lng']
    for n in bad_neighbors:
        if bad_neighbors[n]['rssi'] or bad_neighbors[n]['snr']:
            bad_h = H.get_hotspot_by_addr(n)
            dist_km, heading = utils.haversine_km(
                hlat,
                hlng,
                bad_h['lat'],
                bad_h['lng'],
                return_heading=True
            )
            own = 'same' if hotspot['owner'] == bad_h['owner'] else bad_h['owner'][-5:]
            print(f"{H.get_hotspot_by_addr(n)['name']:29} | {own:5} | {dist_km:5.1f}   | {__heading2str__(heading):7} | {bad_neighbors[n]['rssi']:3d}/{bad_neighbors[n]['ttl']:3d} ({bad_neighbors[n]['rssi']*100/bad_neighbors[n]['ttl']:3.0f}%) | {bad_neighbors[n]['snr']:3d}/{bad_neighbors[n]['ttl']:3d} ({bad_neighbors[n]['snr']*100/bad_neighbors[n]['ttl']:3.0f}%) |")

def map_color_rsrp(rsrp):
    if (int(rsrp) in range(-70, -50)):
        return '#10FF00'
    elif (int(rsrp) in range(-90, -70)):
        return 'green'
    elif (int(rsrp) in range(-110, -90)):
        return 'blue'
    elif (int(rsrp) in range(-130, -110)):
        return '#FF7000'
    else:  # range(-150, -130)
        return 'grey'
    
def poc_polar(hotspot, chals):

    H = Hotspots()
    haddr = hotspot['address']
    hlat, hlng = hotspot['lat'], hotspot['lng']
    hname=hotspot['name']

    if os.path.exists(hname):
        files = glob(hname+'\\*')
        for file in files:
            os.remove(file)
    else:
        os.mkdir(hname)
        
    wl={}#witnesslist
    rl={}#received list of hotspots(hotspot of intereset has been witness to these or received from them)
    c=299792458
    
    for chal in chals:# loop through challenges
        
        for p in chal['path']: #path?
        
            if p['challengee'] == haddr:# handles cases where hotspot of interest is transmitting
                for w in p['witnesses']:#loop through witnesses so we can get rssi at each location challenge received
                    #print('witness',w)
                    lat=H.get_hotspot_by_addr(w['gateway'])['lat']
                    lng=H.get_hotspot_by_addr(w['gateway'])['lng']
                    name=H.get_hotspot_by_addr(w['gateway'])['name']
                    dist_km, heading = utils.haversine_km(hlat,
                                                          hlng,
                                                          lat,
                                                          lng,
                                                          return_heading=True)
                    
                    fspl=20*log10((dist_km+0.01)*1000)+20*log10(915000000)+20*log10(4*pi/c)-27
                    
                    try:
                        wl[w['gateway']]['lat']=lat
                        wl[w['gateway']]['lng']=lng
                        wl[w['gateway']]['rssi'].append(w['signal'])
                    except KeyError:
                        wl[w['gateway']]={'rssi':[w['signal'],],
                                          'dist_km':dist_km,
                                          'heading':heading,
                                          'fspl':fspl,
                                          'lat':lat,
                                          'lng':lng,
                                          'name':name}
            else: # hotspot of interest is not transmitting but may be a witness
                challengee=p['challengee']
                name=H.get_hotspot_by_addr(challengee)['name']
                for w in p['witnesses']:
                    if w['gateway'] != haddr:
                        continue
                    #print('transmitter ', name)
                    #print('witness ', H.get_hotspot_by_addr(w['gateway'])['name']) # hotspot of interest was a witness



                    lat=H.get_hotspot_by_addr(challengee)['lat']
                    lng=H.get_hotspot_by_addr(challengee)['lng']
                    #name=H.get_hotspot_by_addr(w['gateway'])['name']
                    dist_km, heading = utils.haversine_km(hlat,
                                                          hlng,
                                                          lat,
                                                          lng,
                                                          return_heading=True)
                    
                    fspl=20*log10((dist_km+0.01)*1000)+20*log10(915000000)+20*log10(4*pi/c)-27
                    
                    try:
                        rl[challengee]['lat']=lat
                        rl[challengee]['lng']=lng
                        rl[challengee]['rssi'].append(w['signal'])
                    except KeyError:
                        rl[challengee]={'rssi':[w['signal'],],
                                          'dist_km':dist_km,
                                          'heading':heading,
                                          'fspl':fspl,
                                          'lat':lat,
                                          'lng':lng,
                                          'name':name}


    #print('rl:',rl)                    
    ratios=[1.0]*16
    rratios=[1.0]*16
    N=len(ratios)-1
    angles=[]
    rangles=[]
    #angles = [n / float(N) *2 *pi for n in range(N+1)]
    angles = list(np.arange(0.0, 2 * np.pi+(2 * np.pi / N), 2 * np.pi / N))
    rangles=list(np.arange(0.0, 2 * np.pi+(2 * np.pi / N), 2 * np.pi / N))
    #print(angles,len(angles))
    #print(ratios,len(ratios))

    markers=[]
    encoded={}
    rencoded={}
    for w in wl: #for witness in witnesslist
        #print(wl[w])
        mean_rssi=sum(wl[w]['rssi'])/len(wl[w]['rssi'])
        ratio=wl[w]['fspl']/mean_rssi*(-1)
        if ratio > 3.0:
            ratio=3.0
        ratios.append(ratio)
        angles.append(wl[w]['heading']*pi/180)

        #markers.append(folium.Marker([wl[w]['lat'],wl[w]['lng']],popup=wl[w]['name']))
        markers.append([[wl[w]['lat'],wl[w]['lng']],wl[w]['name']])

        # the histogram of the data
        #unique=set(wl[w]['rssi'])
        #num_unique=len(unique)
        
        n, bins, patches = plt.hist(wl[w]['rssi'], 10)#, density=True, facecolor='g', alpha=0.75,)
        plt.xlabel('RSSI(dB)')
        plt.ylabel('Count(Number of Packets)')
        wit=str(wl[w]['name'])
        plt.title('Packets from '+hname+' measured at '+wit)
        #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        #plt.xlim(40, 160)
        #plt.ylim(0, 0.03)
        plt.grid(True)
        #plt.show()
        strFile=str(wl[w]['name'])+'.jpg'
        strWitness=str(wl[w]['name'])
        
        if os.path.isfile(strFile):
            #print('remove')
            os.remove(strFile)   # Opt.: os.system("rm "+strFile)
        plt.savefig(hname+'//'+strFile)
        encoded[strWitness] = base64.b64encode(open(hname+'//'+strFile, 'rb').read())
        plt.close()

    for w in rl: #for witness in witnesslist
        #print(rl[w])
        mean_rssi=sum(rl[w]['rssi'])/len(rl[w]['rssi'])
        rratio=rl[w]['fspl']/mean_rssi*(-1)
        if rratio > 3.0:
            rratio=3.0
        rratios.append(rratio)
        rangles.append(rl[w]['heading']*pi/180)

        #markers.append([[wl[w]['lat'],wl[w]['lng']],wl[w]['name']])
        
        n, bins, patches = plt.hist(rl[w]['rssi'], 10)#, density=True, facecolor='g', alpha=0.75,)
        plt.xlabel('RSSI(dB)')
        plt.ylabel('Count(Number of Packets)')
        wit=str(rl[w]['name'])
        plt.title('Packets from '+wit+' measured at '+hname)

        plt.grid(True)
        #plt.show()
        strFile='rrr'+str(rl[w]['name'])+'.jpg'
        strWitness=str(rl[w]['name'])
        
        if os.path.isfile(strFile):
            #print('remove')
            os.remove(strFile)   # Opt.: os.system("rm "+strFile)
        plt.savefig(hname+'//'+strFile)
        rencoded[strWitness] = base64.b64encode(open(hname+'//'+strFile, 'rb').read())
        plt.close()
    
    # create polar chart
    angles,ratios=zip(*sorted(zip(angles,ratios)))
    rangles,rratios=zip(*sorted(zip(rangles,rratios)))
    angles, ratios = (list(t) for t in zip(*sorted(zip(angles, ratios))))
    rangles, rratios = (list(t) for t in zip(*sorted(zip(rangles, rratios))))

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(angles,ratios, marker='^', linestyle='solid',color='tomato',linewidth=2,markersize=5, label='Transmitting') #markerfacecolor='m', markeredgecolor='k',
    ax.plot(rangles,rratios, marker='v', linestyle='solid',color='dodgerblue',linewidth=1,markersize=5, label='Receiving') #, markerfacecolor='m', markeredgecolor='k'
    ax.legend(bbox_to_anchor=(0,1),fancybox=True, framealpha=0,loc="lower left",facecolor='#000000')
    plt.xlabel('FSPL/RSSI')

    plt.savefig(hname+'//'+hname+'.png',transparent=True)
    #plt.show()

    # add polar chart as a custom icon in map
    m = folium.Map([hlat,hlng], tiles='stamentoner', zoom_start=18,control_scale=True,max_zoom=20)
    polargroup = folium.FeatureGroup(name='Polar Plot')
    
    icon = folium.features.CustomIcon(icon_image=hname+'//'+hotspot['name']+'.png', icon_size=(640,480))
    marker=folium.Marker([hlat,hlng],
              popup=hotspot['name'],
              icon=icon)
    polargroup.add_child(marker)

    # add witness markers
    hsgroup = folium.FeatureGroup(name='Witnesses')
    hsgroup.add_child(folium.Marker([hlat,hlng],popup=hotspot['name']))
    # add the witness markers
    for marker in markers:
        #html = '<img src="data:image/jpg;base64,{}">'.format
        html= '<p><img src="data:image/jpg;base64,{}" alt="" width=640 height=480 /></p> \
               <p><img src="data:image/jpg;base64,{}" alt="" width=640 height=480 /></p>'.format
        
        #print('marker',marker)
        iframe = IFrame(html(encoded[marker[1]].decode('UTF-8'),rencoded[marker[1]].decode('UTF-8')), width=640+25, height=960+40)
        popup = folium.Popup(iframe, max_width=2650)
        
        mark=folium.Marker(marker[0],
                        popup=popup)

        hsgroup.add_child(mark)

    radius=0.01
    center = Point(hlat,hlng)          
    circle = center.buffer(radius)  # Degrees Radius
    gjcircle=shapely.geometry.mapping(circle)
    circle = center.buffer(radius*25)  # Degrees Radius
    gjcircle8=shapely.geometry.mapping(circle)
    
    dcgroup = folium.FeatureGroup(name='Distance Circles',show=False)
    radius=0.01
    center = Point(hlat,hlng)          
    circle = center.buffer(radius)  # Degrees Radius
    gjcircle=shapely.geometry.mapping(circle)
    circle=gjcircle['coordinates'][0]
    my_Circle=folium.Circle(location=[hlat,hlng], radius=300, popup='300m', tooltip='300m')
    dcgroup.add_child(my_Circle)
    my_Circle=folium.Circle(location=[hlat,hlng], radius=1000, popup='1km', tooltip='1km')
    dcgroup.add_child(my_Circle)
    my_Circle=folium.Circle(location=[hlat,hlng], radius=2000, popup='2km', tooltip='2km')
    dcgroup.add_child(my_Circle)
    my_Circle=folium.Circle(location=[hlat,hlng], radius=3000, name='circles',popup='3km', tooltip='3km')
    dcgroup.add_child(my_Circle)
    my_Circle=folium.Circle(location=[hlat,hlng], radius=4000, popup='4km', tooltip='4km')
    dcgroup.add_child(my_Circle)
    my_Circle=folium.Circle(location=[hlat,hlng], radius=5000, popup='5km', tooltip='5km')
    dcgroup.add_child(my_Circle)
    my_Circle=folium.Circle(location=[hlat,hlng], radius=10000, popup='10km', tooltip='10km')
    dcgroup.add_child(my_Circle)

    h3colorgroup = folium.FeatureGroup(name='h3 Hexagon Grid Color Fill',show=False)
    style = {'fillColor': '#f5f5f5', 'lineColor': '#ffffbf'}    
    #polygon = folium.GeoJson(gjson, style_function = lambda x: style).add_to(m)

    h3group = folium.FeatureGroup(name='h3 r11 Hex Grid',show=False)
    h3namegroup = folium.FeatureGroup(name='h3 r11 Hex Grid Names',show=False)
    h3fillgroup = folium.FeatureGroup(name='h3 r11 Hex Grid Color Fill',show=True)
    h3r8namegroup = folium.FeatureGroup(name='h3 r8 Hex Grid Names',show=False)
    h3r8group = folium.FeatureGroup(name='h3 r8 Hex Grid',show=False)
    hexagons = list(h3.polyfill(gjcircle, 11))
    hexagons8 = list(h3.polyfill(gjcircle8, 8))
    polylines = []
    
    lat = []
    lng = []
    i=0
    #print('hexagon',hexagons[0])
    #print(dir(h3))
    home_hex=h3.geo_to_h3(hlat,hlng,11)
    a=h3.k_ring(home_hex,7)
    for h in a:
        gjhex=h3.h3_to_geo_boundary(h,geo_json=True)
        gjhex=geometry.Polygon(gjhex)
        mean_rsrp=-60
        folium.GeoJson(gjhex,
                   style_function=lambda x, mean_rsrp=mean_rsrp: {
                   'fillColor': map_color_rsrp(mean_rsrp),
                   'color': map_color_rsrp(mean_rsrp),
                   'weight': 1,
                   'fillOpacity': 0.5},
                   #tooltip='tooltip'
                   ).add_to(h3fillgroup)
    
    for hex in hexagons:
        p2=h3.h3_to_geo(hex)
        #p2 = [45.3311, -121.7113]
        folium.Marker(p2, name='hex_names',icon=DivIcon(
                #icon_size=(150,36),
                #icon_anchor=(35,-45),
                icon_anchor=(35,0),
                html='<div style="font-size: 6pt; color : black">'+str(hex)+'</div>',
                )).add_to(h3namegroup)
        #m.add_child(folium.CircleMarker(p2, radius=15))
    
        polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
        # flatten polygons into loops.
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        lat.extend(map(lambda v:v[0],polyline))
        lng.extend(map(lambda v:v[1],polyline))
        polylines.append(polyline)
        
    for polyline in polylines:
        my_PolyLine=folium.PolyLine(locations=polyline,weight=1,color='blue')
        h3group.add_child(my_PolyLine)



    polylines = []
    
    lat = []
    lng = []
    #polylines8 = []
    for hex in hexagons8:
        p2=h3.h3_to_geo(hex)
        folium.Marker(p2, name='hex_names',icon=DivIcon(
                #icon_size=(150,36),
                #icon_anchor=(35,-45),
                icon_anchor=(35,0),
                html='<div style="font-size: 8pt; color : black">'+str(hex)+'</div>',
                )).add_to(h3r8namegroup)
    
        polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
        # flatten polygons into loops.
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        lat.extend(map(lambda v:v[0],polyline))
        lng.extend(map(lambda v:v[1],polyline))
        polylines.append(polyline)
        
    for polyline in polylines:
        my_PolyLine=folium.PolyLine(locations=polyline,weight=1,color='blue')
        h3r8group.add_child(my_PolyLine)

    # add possible tiles
    folium.TileLayer('cartodbpositron').add_to(m)
    folium.TileLayer('cartodbdark_matter').add_to(m)
    folium.TileLayer('openstreetmap').add_to(m)
    folium.TileLayer('Mapbox Bright').add_to(m)
    #folium.TileLayer('stamentoner').add_to(m)

    # add markers layer
    #marker_cluster = MarkerCluster().add_to(m)

    polargroup.add_to(m)#polar plot
    hsgroup.add_to(m)#hotspots
    dcgroup.add_to(m)#distance circles
    h3group.add_to(m)
    h3namegroup.add_to(m)
    h3fillgroup.add_to(m)
    m.keep_in_front(h3group)
    h3r8group.add_to(m)
    h3r8namegroup.add_to(m)

    # add the layer control
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(hname+'//'+hname+'_map.html')


def poc_reliability(hotspot, challenges):
    """

    :param hotspot:
    :param challenges: list of challenges
    :return:
    """
    H = Hotspots()
    haddr = hotspot['address']

    # iterate through challenges finding actual interactions with this hotspot
    results_tx = dict()  # key = tx addr, value = [pass, fail]
    results_rx = dict()  # key = rx addr, value = [pass, fail]
    for chal in challenges:
        pnext = chal['path'][-1]
        pnext_pass = pnext['witnesses'] or pnext['receipt']

        for p in chal['path'][:-1][::-1]:
            if pnext_pass or p['witnesses'] or p['receipt']:
                if pnext['challengee'] == haddr:
                    if p['challengee'] not in results_rx:
                        results_rx[p['challengee']] = [0, 0]
                    results_rx[p['challengee']][0 if pnext_pass else 1] += 1
                if p['challengee'] == haddr:
                    if pnext['challengee'] not in results_tx:
                        results_tx[pnext['challengee']] = [0, 0]
                    results_tx[pnext['challengee']][0 if pnext_pass else 1] += 1
                pnext_pass = True
            pnext = p

    hlat = hotspot['lat']
    hlon = hotspot['lng']

    def summary_table(results, hotspot_transmitting=False):

        other_pass = 0
        other_ttl = 0
        other_cnt = 0
        all_ttl = 0
        all_pass = 0
        dist_min = 9999
        dist_max = 0

        if hotspot_transmitting:
            print(f"PoC hops from: {hotspot['name']}")
            print(f"{'to receiving hotspot':30} | owner | {'dist km'} | {'heading'} | recv/ttl | recv % |")
        else:
            print(f"PoC hops to: {hotspot['name']}")
            print(f"{'from transmitting hotspot':30} | owner | {'dist km'} | {'heading'} | recv/ttl | recv % |")
        print("-" * 80)

        # print in descending order
        sort_keys = [(results[r][0]+results[r][1], r) for r in results]
        sort_keys.sort(reverse=True)

        for h in [sk[1] for sk in sort_keys]:
            ttl = results[h][0] + results[h][1]
            all_ttl += ttl
            all_pass += results[h][0]
            dist, heading = utils.haversine_km(
                hlat, hlon,
                H.get_hotspot_by_addr(h)['lat'], H.get_hotspot_by_addr(h)['lng'],
                return_heading=True
            )

            heading = 5 * round(heading / 5, 0)
            idx = int(round(heading / 45)) % 8
            headingstr = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            if ttl == 1:
                other_ttl += ttl
                other_pass += results[h][0]
                other_cnt += 1
                dist_min = min(dist_min, dist)
                dist_max = max(dist_max, dist)
                continue
            ownr = 'same' if hotspot['owner'] == H.get_hotspot_by_addr(h)['owner'] else H.get_hotspot_by_addr(h)['owner'][-5:]
            print(f"{H.get_hotspot_by_addr(h)['name']:30} | {ownr:5} | {dist:6.1f}  | {heading:4.0f} {headingstr[idx]:>2} | {results[h][0]:3d}/{ttl:3d}  | {results[h][0] / ttl * 100:5.0f}% |")

        if other_ttl:
            print(f"other ({other_cnt:2}){' ' * 20} |  N/A  | {dist_min:4.1f}-{dist_max:2.0f} |   N/A   | {other_pass:3d}/{other_ttl:3d}  | {other_pass / other_ttl * 100:5.0f}% | ")
        if all_ttl:
            print(f"{' ' * 40}{' ' * 10}         ---------------------")
            print(f"{' ' * 40}{' '*10}   TOTAL | {all_pass:3d}/{all_ttl:4d} | {all_pass / all_ttl * 100:5.0f}% | ")

    summary_table(results_tx, hotspot_transmitting=True)
    print()
    print()
    summary_table(results_rx, hotspot_transmitting=False)

def main():
    parser = argparse.ArgumentParser("analyze hotspots", add_help=True)
    parser.add_argument('-x', help='report to run', choices=['poc_reliability', 'poc_v10', 'poc_polar', 'poc_summary'], required=True)

    parser.add_argument('-c', '--challenges', help='number of challenges to analyze, default:500', default=500, type=int)
    parser.add_argument('-n', '--name', help='hotspot name to analyze with dashes-between-words')
    parser.add_argument('-a', '--address', help='hotspot address to analyze')

    args = parser.parse_args()
    H = Hotspots()
    hotspot = None
    if args.name:
        hotspot = H.get_hotspot_by_name(args.name)
        if hotspot is None:
            raise ValueError(f"could not find hotspot named '{args.name}' use dashes between words")
    elif args.address:
        hotspot = H.get_hotspot_by_addr(args.address)
        if hotspot is None:
            raise ValueError(f"could not find hotspot address '{args.address}' ")
    else:
        raise ValueError("must provide hotspot address '--address' or name '--name'")

    challenges = utils.load_challenges(hotspot['address'], args.challenges)
    challenges = challenges[:args.challenges]
    if len(challenges) < 2:
        print(f"ERROR could not load challenges, either hotspot has been offline too long or you need to increase --challenge arguement")
        return
    days, remainder = divmod(challenges[0]['time'] - challenges[-1]['time'], 3600 * 24)
    hours = int(round(remainder / 3600, 0))
    print(f"analyzing {len(challenges)} challenges from block {challenges[0]['height']}-{challenges[-1]['height']} over {days} days, {hours} hrs")

    if args.x == 'poc_summary':
        poc_summary(hotspot, challenges)
    if args.x == 'poc_reliability':
        poc_reliability(hotspot, challenges)
    if args.x == 'poc_polar':
        poc_polar(hotspot, challenges)
    if args.x == 'poc_v10':
        pocv10_violations(hotspot, challenges)



if __name__ == '__main__':
    main()



