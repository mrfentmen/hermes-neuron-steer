#!/usr/bin/env bash
# NeuronSteer demo v2 — motion cut.
# Generated art under slow zooms/pans (parallax layers move at different
# rates), text sliding in, an animated multiplier bar, a travelling pulse
# on the dose-scan grid, and a terminal sequence. Silent, 1280x720, ~76s.
set -euo pipefail
cd "$(dirname "$0")"
BOLD="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REG="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
[ -f "$MONO" ] || MONO="$REG"
ART1="art/media-generation-neuron-circuit-art-0-c942a256-e6ed-45a8-880e-df67c396a41d.webp"
ART2="art/media-generation-signal-flow-art-0-507ebb6f-c8d1-4e5e-94d3-af428945051e.webp"
mkdir -p v2 txt2

T() { printf '%s' "$2" > "txt2/$1.txt"; }
T s1t "NeuronSteer";            T s1s "Neuron steering for Hermes Agent"
T s2t "Nous Research, productised"
T s2s "Their method: Contrastive Neuron Attribution, arXiv 2605.12290"
T s3h "1. FIND the circuit"
T l1 '$ hermes steer find --name refusal'
T l2 'positive prompts: requests that trigger the behavior'
T l3 'negative prompts: matched requests that do not'
T l4 'recording MLP activations at the last token...'
T l5 'ranking every neuron by mean activation difference'
T l6 'top 0.1 percent saved as a named circuit'
T s4h "2. STEER with it"
T m1 "multiplier x1.0  —  baseline"
T m2 "multiplier x0.0  —  circuit ablated"
T m3 "multiplier x2.0  —  circuit amplified"
T s5h "3. SCAN the dose window"
T s5s "Where the behavior turns on, before output quality breaks."
T s6h "Four tools for the agent"
T t1 "neuron_steer_find"
T t2 "neuron_steer_generate"
T t3 "neuron_steer_scan"
T t4 "neuron_steer_list"
T s6s "plus a /steer command and the hermes steer CLI"
T s7t "13 of 13 tests passing"
T s7s "github.com/Mrfentmen/hermes-neuron-steer"

# --- Scene 1: title over the circuit art, slow zoom in -------------------
ffmpeg -y -v error -i "$ART1" -vf "\
scale=2560:1440,zoompan=z='1+0.13*on/269':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=270:s=1280x720:fps=30,\
eq=brightness=-0.10,\
drawtext=fontfile=${BOLD}:textfile=txt2/s1t.txt:fontcolor=white:fontsize=84:x=(w-text_w)/2:y='if(lt(t,1.1),820-470*t,300)',\
drawtext=fontfile=${REG}:textfile=txt2/s1s.txt:fontcolor=0xd0d7de:fontsize=36:x=(w-text_w)/2:y='if(lt(t,1.4),900-460*t,410)'" \
  -c:v libx264 -pix_fmt yuv420p v2/01.mp4

# --- Scene 2: method over the signal art, slow pan ------------------------
ffmpeg -y -v error -i "$ART2" -vf "\
scale=2560:1440,zoompan=z='1.25':x='(iw-iw/zoom)*on/299':y='(ih-ih/zoom)/2':d=300:s=1280x720:fps=30,\
eq=brightness=-0.16,\
drawtext=fontfile=${BOLD}:textfile=txt2/s2t.txt:fontcolor=white:fontsize=56:x='if(lt(t,1),-700+760*t,60)':y=290,\
drawtext=fontfile=${REG}:textfile=txt2/s2s.txt:fontcolor=0xd0d7de:fontsize=33:x='if(lt(t,1.3),-760+760*t,60)':y=380" \
  -c:v libx264 -pix_fmt yuv420p v2/02.mp4

# --- Scene 3: FIND, terminal lines appearing over dimmed circuit art ------
ffmpeg -y -v error -i "$ART1" -vf "\
scale=2560:1440,zoompan=z='1.10+0.08*on/389':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=390:s=1280x720:fps=30,\
eq=brightness=-0.38,\
drawtext=fontfile=${BOLD}:textfile=txt2/s3h.txt:fontcolor=white:fontsize=54:x=70:y=110,\
drawtext=fontfile=${MONO}:textfile=txt2/l1.txt:fontcolor=0x7ee787:fontsize=29:x=80:y=250:enable='gte(t,0.8)',\
drawtext=fontfile=${MONO}:textfile=txt2/l2.txt:fontcolor=0xd0d7de:fontsize=29:x=80:y=300:enable='gte(t,2.6)',\
drawtext=fontfile=${MONO}:textfile=txt2/l3.txt:fontcolor=0xd0d7de:fontsize=29:x=80:y=350:enable='gte(t,4.4)',\
drawtext=fontfile=${MONO}:textfile=txt2/l4.txt:fontcolor=0xd0d7de:fontsize=29:x=80:y=400:enable='gte(t,6.2)',\
drawtext=fontfile=${MONO}:textfile=txt2/l5.txt:fontcolor=0xd0d7de:fontsize=29:x=80:y=450:enable='gte(t,8.0)',\
drawtext=fontfile=${MONO}:textfile=txt2/l6.txt:fontcolor=0xffd166:fontsize=29:x=80:y=500:enable='gte(t,9.8)',\
drawbox=x=82:y=552:w=16:h=28:color=0x7ee787:t=fill:enable='lt(mod(t,1),0.55)'" \
  -c:v libx264 -pix_fmt yuv420p v2/03.mp4

# --- Scene 4: STEER, segmented level meter --------------------------------
# Fixed-geometry segments switched by enable windows only: this ffmpeg
# build does not animate drawbox w/x/y expressions, but enable windows
# evaluate per frame (proven by the text windows in this same scene).
SEG=""
for p in 0 1 2 3 4 5 6 7 8 9; do
  X=$((140 + p * 100))
  OFF=$(awk "BEGIN{printf \"%.2f\", 4 + 0.4 * (10 - $p) - 0.01}")
  ON=$(awk "BEGIN{printf \"%.2f\", 8 + 0.35 * $p + 0.01}")
  SEG="${SEG}drawbox=x=${X}:y=430:w=92:h=22:color=0x3fb950:t=fill:enable='between(t,4,${OFF})',"
  SEG="${SEG}drawbox=x=${X}:y=430:w=92:h=22:color=0x3fb950:t=fill:enable='gte(t,${ON})',"
done
for p in 8 9; do
  X=$((140 + p * 100))
  ON=$(awk "BEGIN{printf \"%.2f\", 8 + 0.35 * $p + 0.01}")
  SEG="${SEG}drawbox=x=${X}:y=430:w=92:h=22:color=0xffd166:t=fill:enable='gte(t,${ON})',"
done
ffmpeg -y -v error -i "$ART2" -vf "\
scale=2560:1440,zoompan=z='1.30-0.12*on/359':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=360:s=1280x720:fps=30,\
eq=brightness=-0.30,\
drawtext=fontfile=${BOLD}:textfile=txt2/s4h.txt:fontcolor=white:fontsize=54:x=70:y=110,\
drawbox=x=140:y=430:w=1000:h=22:color=0x30363d:t=fill,\
drawbox=x=140:y=430:w=1000:h=22:color=0x3fb950:t=fill:enable='between(t,0.3,3.99)',\
${SEG}\
drawtext=fontfile=${MONO}:textfile=txt2/m1.txt:fontcolor=white:fontsize=34:x=(w-text_w)/2:y=490:enable='between(t,0.3,3.99)',\
drawtext=fontfile=${MONO}:textfile=txt2/m2.txt:fontcolor=white:fontsize=34:x=(w-text_w)/2:y=490:enable='between(t,4,7.99)',\
drawtext=fontfile=${MONO}:textfile=txt2/m3.txt:fontcolor=white:fontsize=34:x=(w-text_w)/2:y=490:enable='gte(t,8)'" \
  -c:v libx264 -pix_fmt yuv420p v2/04.mp4

# --- Scene 5: SCAN, pulse travelling a sine path over a grid --------------
# The moving dots are separate color inputs moved with overlay, whose
# x/y expressions do evaluate per frame (drawbox x/y do not, here).
ffmpeg -y -v error \
  -f lavfi -i "color=c=0x0d1117:s=1280x720:r=30:d=12" \
  -f lavfi -i "color=c=0xffd166:s=16x16:r=30:d=12" \
  -f lavfi -i "color=c=0x3fb950:s=9x9:r=30:d=12" \
  -filter_complex "\
[0:v]drawgrid=width=64:height=64:thickness=1:color=0x16324f,\
drawtext=fontfile=${BOLD}:textfile=txt2/s5h.txt:fontcolor=white:fontsize=54:x=70:y=110,\
drawtext=fontfile=${REG}:textfile=txt2/s5s.txt:fontcolor=0xd0d7de:fontsize=33:x=70:y=580[bg];\
[bg][1:v]overlay=x='60+1160*t/12':y='350+110*sin(2*PI*t/2.5)'[o1];\
[o1][2:v]overlay=x='60+1160*(t-0.35)/12':y='354+110*sin(2*PI*(t-0.35)/2.5)':enable='gte(t,0.4)'[out]" \
  -map "[out]" -c:v libx264 -pix_fmt yuv420p v2/05.mp4

# --- Scene 6: tools sliding in one by one ---------------------------------
ffmpeg -y -v error -f lavfi -i "color=c=0x0d1117:s=1280x720:r=30:d=11" -vf "\
drawbox=x=0:y=0:w=1280:h=6:color=0x3fb950:t=fill,\
drawtext=fontfile=${BOLD}:textfile=txt2/s6h.txt:fontcolor=white:fontsize=54:x=70:y=100,\
drawtext=fontfile=${MONO}:textfile=txt2/t1.txt:fontcolor=0x7ee787:fontsize=36:x='if(lt(t,0.7),-600+700*t,100)':y=230,\
drawtext=fontfile=${MONO}:textfile=txt2/t2.txt:fontcolor=0x7ee787:fontsize=36:x='if(lt(t,2.2),-700+700*(t-1.5),100)':y=310:enable='gte(t,1.5)',\
drawtext=fontfile=${MONO}:textfile=txt2/t3.txt:fontcolor=0x7ee787:fontsize=36:x='if(lt(t,3.7),-600+700*(t-3),100)':y=390:enable='gte(t,3)',\
drawtext=fontfile=${MONO}:textfile=txt2/t4.txt:fontcolor=0x7ee787:fontsize=36:x='if(lt(t,5.2),-600+700*(t-4.5),100)':y=470:enable='gte(t,4.5)',\
drawtext=fontfile=${REG}:textfile=txt2/s6s.txt:fontcolor=0xd0d7de:fontsize=31:x=100:y=580:enable='gte(t,6)'" \
  -c:v libx264 -pix_fmt yuv420p v2/06.mp4

# --- Scene 7: close over the circuit art, slow zoom out --------------------
ffmpeg -y -v error -i "$ART1" -vf "\
scale=2560:1440,zoompan=z='1.16-0.14*on/269':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=270:s=1280x720:fps=30,\
eq=brightness=-0.10,\
drawtext=fontfile=${BOLD}:textfile=txt2/s7t.txt:fontcolor=white:fontsize=58:x=(w-text_w)/2:y=290,\
drawtext=fontfile=${REG}:textfile=txt2/s7s.txt:fontcolor=0xffd166:fontsize=34:x=(w-text_w)/2:y=390" \
  -c:v libx264 -pix_fmt yuv420p v2/07.mp4

printf "file '%s'\n" 01.mp4 02.mp4 03.mp4 04.mp4 05.mp4 06.mp4 07.mp4 > v2/list.txt
ffmpeg -y -v error -f concat -safe 0 -i v2/list.txt -c copy neuronsteer-demo-v2.mp4
ls -la neuronsteer-demo-v2.mp4
