#GUI 코드 (tkinter py): 김현주, 강호준, 최정식

from tkinter import *
import tkinter
import urllib
import urllib.request
import tkinter.messagebox
from datetime import datetime
import pandas as pd
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import numpy as np
import matplotlib.pyplot as plt

root = tkinter.Tk()
root.wm_title("SMART LIFE")
root.geometry("700x700+100+100")

#현재 시간 표시
time_string = time.strftime('%H:%M:%S')
label = tkinter.Label(root, text = "현재 시간 " + time_string, fg="blue")
label.pack()

#대교 위치 - 버튼 눌렀을때 이미지 변경
label2 = tkinter.Label(root, text="버튼을 눌러 위치와 주변시설 보기")
label2.pack()  # 위젯 배치

#button 눌렀을 때 지도 이미지 변경하기
def loc_hangang():

   img1 = PhotoImage(file='C:\\Users\\judy5\\PycharmProjects\\waterlevel_p\\hangang.png')
   img1 = img1.zoom(1)
   panel.configure(image=img1)
   panel.image = img1

def loc_jamsu():
   img2 = PhotoImage(file='C:\\Users\\judy5\\PycharmProjects\\waterlevel_p\\jamsu.png')
   panel.configure(image=img2)
   panel.image = img2

def loc_cheongdam():
   img3 = PhotoImage(file='C:\\Users\\judy5\\PycharmProjects\\waterlevel_p\\cheongdam.png')
   panel.configure(image=img3)
   panel.image = img3


def info_hangang():
    window = tkinter.Tk()

    window.title("hangang")
    window.geometry("700x300")

    label1 = tkinter.Label(window, text = "한강대교 주변 시설\n", font = (20))

    label2 = tkinter.Label(window, text = "1. 한강 자전거길 \n"
                                          "\n"
                                          "자전거를 한강을 조망하면서 탈 수 있습니다.\n")

    label3 = tkinter.Label(window, text = "2. 노들섬 \n"
                                          "\n"
                                          "1950년대까지 중지도로 불리며 백사장과 스케이트장으로 활용되었던 노들섬은 6-70년대 한강개발계획으로\n"
                                          "중지도의 모래를 사용하면서 한강중앙에 떠있는 섬이 되었다. \n"
                                          "이와 함께 약 50여년간 제대로 활용되지 못하다가 2005년 서울시가 섬을 매입하고 시민들의문화 휴식 공간으로 만들기위한\n"
                                          "준비를 시작하면서 다양한 시도 끝에 2019년 9월 최대한 원형을 그대로 간직한 채로 시민들을 위한 복합문화공간으로 돌아왔다. \n"
                                          "노들섬의 대표적 문화시설로는 라이브하우스, 노들서가, 앤테이블, 뮤직라운지류, 식물도, 스페이스445,\n"
                                          "다목적홀숲 등이 있으며 각 시설들을 중심으로 다양한 문화 프로그램이 진행된다.")


    label1.pack()
    label2.pack()
    label3.pack()

    window.mainloop()

def info_jamsu():
    window = tkinter.Tk()

    window.title("jamsu")
    window.geometry("700x600")

    label1 = tkinter.Label(window, text = "잠수교 주변 시설\n", font = (20))


    label2 = tkinter.Label(window, text = "1. 한강 자전거길 \n"
                                          "\n"
                                          "자전거를 한강을 조망하면서 탈 수 있습니다.\n")

    label3 = tkinter.Label(window, text = "2. 반표한강공원 \n"
                                          "\n"
                                          "물방울놀이터, 인라인 전용트랙, 축구장, 농구장 등의 체육시설과 더불어 동작대교 남단에 설치된\n"
                                          "전망대인 노을카페와 구름카페에서 탁 트인 한강수면과 강변 빌딩숲을 감상할 수 있다.\n"
                                          "그리고 반포한강공원과 연결된 서래섬(盤浦島)은 도심 속 휴식과 놀이공간을 제공하는 인공섬으로\n"
                                          "반포대교와 동작대교 사이에 위치해 있다.\n"
                                          "봄이 되면 노란 유채꽃이 만발하여 '서래섬 나비ㆍ유채꽃 축제' 로 나들이와 산책을 즐기는 사람들이 많다.\n"
                                          "이 외에도 반포한강공원에는 생태학습장, 보트장, 자전거도로, 피크닉장 등 볼거리와 즐길거리가 무궁무진하다.\n")


    label4 = tkinter.Label(window, text = "3. 달빛 무지개 분수 \n"
                                          "\n"
                                          "서울 반포대교에 설치된 분수인 달빛무지개분수는 반포대교 570m 구간 양측 총 1천140m에 380개 노즐을 설치해\n"
                                          "수중펌프로 끌어올린 한강물을 약 20m 아래 한강 수면으로 떨어뜨리는 새로운 개념의 분수이다.\n"
                                          "달빛 무지개 분수는 낮과 밤에 다른 모습을 즐길 수 있다.\n"
                                          "뿜어내는 물의 양만 분당 190톤에 달하는 달빛 무지개 분수는 낮에는 분수에 떨어지는 물결의 모양에 따라 휘날리는\n"
                                          "버들가지와 버들잎을 형상화한 모양 등 백 여 가지의 다양한 모습의 분수를 만들어낸다.\n"
                                          "밤에는 긍정과 희망의 이미지를 상징하는 무지개 색깔의 분수로 화려하게 변신한다.\n"
                                          "설치된 조명 200개는 아름다운 무지개 모양의 야경을 선사하고 음악에 맞춰 춤추는 분수는 시민들에게 색다른 즐거움을 선사한다.\n"
                                          "또한 달빛 무지개 분수는 지난 2008년 12월 세계 최장 교량 분수로 기네스북에 등재되면서 최고의 위용을 세계에서도 인정받았다.\n"
                                          "이 분수는 매년 4월부터 10월까지 매일 가동되며 하루 4~6회(회당 20분씩) 가동된다.\n")


    label5 = tkinter.Label(window, text = "4. 새빛섬 \n"
                                          "\n"
                                          "한글 명칭의 '세빛'은 서로 그 빛을 겹칠 때 가장 많은 색깔을 만들어내는 빛의 삼원색 빨강·파랑·초록처럼 3개의 섬이 조화를 이루어\n"
                                          "한강과 서울을 빛내라는 바람을 담고 있고, '둥둥'은 수상에 띄워진 문화공간을 강조하는 의미를 담고 있다.\n"
                                          "반포대교 남단의 한강 수상에 띄운 부체(浮體) 위에 건물을 지어 도교로 연결한 3개의 섬과 별도로 조성된\n"
                                          "미디어아트갤러리로 이루어져 있으며, 건축연면적은 9995㎡이다.\n"
                                          "인공위성 좌표에 따라 인공섬의 윈치(winch)가 와이어를 풀었다 당겼다 하면서 위치를 고정시키고,\n"
                                          "수위가 상승하면 계류체인이 풀리면서 수위를 따라 이동하도록 되어 있다.\n"
                                          "3개의 섬은 제1섬(비스타), 제2섬(비바), 제3섬(테라)으로 구분한다.\n"
                                          "활짝 핀 꽃을 형상화한 제1섬은 건축연면적 5490㎡에 3층으로 이루어져 있으며, 국제회의·리셉션·제작발표회·마케팅 이벤트 등\n"
                                          "다양한 행사를 할 수 있는 컨벤션홀과 레스토랑 등의 부대시설을 갖추고 있다.\n"
                                          "꽃봉오리를 형상화한 제2섬은 건축연면적 3426㎡에 3층으로 이루어져 있으며,\n"
                                          "공연·전시 등의 문화체험 행사와 콘퍼런스·세미나 등의 행사를 유치하는 공간으로 활용된다.\n"
                                          "씨앗을 형상화한 제3섬은 건축연면적 1078㎡에 2층으로 이루어져 있으며, 수상레포츠 공간으로 활용된다.\n"
                                          "이밖에 초대형 LED와 수상무대를 갖춘 미디어아트갤러리는 각종 행사 및 문화예술공간으로 활용된다.")

    label1.pack()
    label2.pack()
    label3.pack()
    label4.pack()
    label5.pack()

    window.mainloop()

def info_cheongdam():
    window = tkinter.Tk()

    window.title("cheongdam")
    window.geometry("700x300")

    label1 = tkinter.Label(window, text="청담대교 주변 시설\n", font = 20)

    label2 = tkinter.Label(window, text="1. 한강 자전거길 \n"
                                        "\n"
                                        "자전거를 한강을 조망하면서 탈 수 있습니다.\n")

    label3 = tkinter.Label(window, text="2.뚝섬 한강 공원 (뚝섬 유원지)\n"
                                        "\n"
                                        "뚝섬한강공원은 한강공원으로 새단장하기 이전부터 강변유원지로 유명했던 곳이다.\n"
                                        "공원 내에는 수변광장, 장미정원, 자연학습장, 어린이 놀이터 등으로 조성되어 있다.\n"
                                        "뚝섬한강공원에는 몸은 가늘고 긴 원통형인 '자벌레' 형태의 길이 243m 규모의 예술과 휴식이 함께하는 복합문화공간이 있다.\n"
                                        "뚝섬유원지역에서 연결되어 있어 누구나 이곳을 통해 편리하게 공원으로 진입할 수 있으며 '자벌레' 통로에는 카페, 찻집,\n"
                                        "기프트숍은 물론 미디어아트 작가들의 작품전시 감상을 할 수 있다.\n"
                                        "여름에는 시원한 바람을 맞으며 낭만과 젊음을 만끽할 수 있는 수상스키, 모터보트 등 수상스포츠가 활발하게 이루어진다.\n"
                                        "봄과 가을에는 카페테리아와 계절 꽃전시장으로, 겨울철에는 눈썰매장과 스케이트장으로 \n"
                                        "사계절 내내 시민들에게 보다 많은 볼거리와 즐길거리를 제공한다.\n"
                                        "뚝섬한강공원에는 이 외에도 X게임장, 인공암벽장, 유람선 선착장, 토요 나눔장터 운영, 수유실,\n"
                                        "여성전용쉼터 등의 시설들로 시민들의 많은 사랑을 받고 있다.")

    label1.pack()
    label2.pack()
    label3.pack()

    window.mainloop()


img0 = PhotoImage(file="C:\\Users\\judy5\\PycharmProjects\\waterlevel_p\\basic.png")
panel = tkinter.Label(root, image=img0)
#panel.pack(side="top", fill="both", anchor=SW, expand="yes")
panel.pack(side="top", fill="none", expand="yes")

button1 = Button(root, text = "한강대교", command = loc_hangang, bg='skyblue')
button2 = Button(root, text = "잠수교", command = loc_jamsu, bg='orange')
button3 = Button(root, text = "청담대교", command = loc_cheongdam, bg='pale green')
button6 = Button(root, text = "한강대교 주변 시설", command = info_hangang, bg='skyblue')
button7 = Button(root, text = "잠수교 주변 시설", command = info_jamsu, bg='orange')
button8 = Button(root, text = "청담대교 주변 시설", command = info_cheongdam, bg='pale green')

button1.place(x=50, y=50)
button2.place(x=120, y=50)
button3.place(x=180, y=50)
button6.place(x=50, y=320)
button7.place(x=170, y=320)
button8.place(x=280, y=320)

#button1.pack(side = BOTTOM, anchor=NW, expand="yes")
#button2.pack(side = BOTTOM, anchor=NW, expand="yes")
#button3.pack(side = BOTTOM, anchor=NW, expand="yes")
#button6.pack(side = BOTTOM, anchor=NW, expand="yes")
#button7.pack(side = BOTTOM, anchor=NW, expand="yes")
#button8.pack(side = BOTTOM, anchor=NW, expand="yes")

label3 = tkinter.Label(root, text="예측 수위 데이터")
label3.pack()  # 위젯 배치

#그래프 그리기
#fig = Figure(figsize=(15, 6), dpi=100)

#realtime_dataload.ipynb
now = datetime.now()
timestamp = datetime.timestamp(now)
y_timestamp = timestamp - 86400
yday = datetime.fromtimestamp(y_timestamp)
yday = str(yday)
yday = yday[:10]
now = str(now)
today = now[0:10]
totime = now[11:13]
tomin = now[14:16]

if int(tomin) >= 30:
    tomin = '00'
else:
    tomin = '30'
    totime = int(totime) - 1

n_features = 3

Obscd = [1018683, 1018680, 1018662]
name = ['hg', 'js', 'cd']

for i in range(len(Obscd)):
    url = 'http://hrfco.go.kr/servlet/sumun/wlExcelDownload.do?Obscd={}&Sdt={}%20{}:{}&Edt={}%20{}:{}&Type=10M'.format(Obscd[i], yday, totime, tomin, today, totime, tomin)
    urllib.request.urlretrieve(url, "{}_{}_rt_data.xls".format(i, name[i]))
    fn = '*_rt_data*.xls'
    fn = ['0_hg_rt_data.xls', '1_js_rt_data.xls', '2_cd_rt_data.xls']

dfs0 = pd.read_html(fn[0], encoding='euc-kr')
dfs1 = pd.read_html(fn[1], encoding='euc-kr')
dfs2 = pd.read_html(fn[2], encoding='euc-kr')

df0 = dfs0[0]
df1 = dfs1[0]
df2 = dfs2[0]

time = df0.iloc[:, 0]
rt_data0 = df0.iloc[:, 1]
rt_data1 = df1.iloc[:, 1]
rt_data2 = df2.iloc[:, 1]

y_show0 = rt_data0
y_show1 = rt_data1
y_show2 = rt_data2

'''
fig.add_subplot(111).plot(y_show0)
fig.add_subplot(111).plot(y_show1)
fig.add_subplot(111).plot(y_show2)

fig.legend(['hangang', 'jamsu', 'chungdam'])
'''


#--------------------------------------------------------------------------------------------
df = pd.read_csv('./add_timestamp_data.csv')
data = df.iloc[:, 2:5]
std1 = data.std()
mean1 = data.mean()

def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict, batch_size=1):
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)

    # The states must be a list
    if not isinstance(states, list):
        states = [states]

    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0], 1, 3))
    print(decoder_input.shape)

    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
            [decoder_input] + states, batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)


from keras.models import load_model

fn_model = 'encoder_hello_generator.h5'
encoder_predict_model = load_model(fn_model)
fn_model = 'decoder_hello_generator.h5'
decoder_predict_model = load_model(fn_model)

fn_realtime='test_20200623_150651.csv'
df = pd.read_csv(fn_realtime)
df = df.iloc[:, 1:4]
dataRecent = np.array(df)
type(dataRecent)
dataf = dataRecent

num_steps_to_predict = 36*2
offset = 50
vdata_in = np.expand_dims(dataf[offset:offset+144], 0)
vdata_out = np.expand_dims(dataf[offset+144:offset+144+num_steps_to_predict], 0)
np.shape(vdata_in),np.shape(vdata_out)
print(len(np.shape(vdata_in)))

if len(np.shape(vdata_in)) == 2:
    x_test = np.expand_dims(vdata_in, axis=0)
else:
    x_test = np.array(vdata_in)

y_test_predicted = predict(x_test, encoder_predict_model, decoder_predict_model, num_steps_to_predict)
print(y_test_predicted.shape)

vdata_in=x_test[0]
y_test_predicted = y_test_predicted[0]

t = np.arange(144+num_steps_to_predict)
n_steps_in = 144
t_in = np.arange(n_steps_in)
t_predict = t[n_steps_in:]-1
figsize = (10,5)
print(t_in)

fig = Figure(figsize=(15, 6), dpi=100)
#plt.figure(figsize=(10,5))
clr= ['C0', 'C1', 'C2']
for i in range(3):
    #plt.plot
    fig.add_subplot(111).plot(t_in, vdata_in[:,i]* std1[i] + mean1[i],color=clr[i])
    #fig.add_subplot(111).plot(t_predict, vdata_out[0][:,i]* std1[i] + mean1[i], color=clr[i])
    #fig.add_subplot(111).plot(t_predict, y_test_predicted[:,i]* std1[i] + mean1[i],color=clr[i], linestyle =':')

for i in range(3):
    fig.add_subplot(111).plot(t_predict, y_test_predicted[:,i]* std1[i] + mean1[i],color=clr[i], linestyle =':')
    fig.legend(['hangang', 'jamsu', 'chungdam'])
    fig.add_subplot(111).set_ylabel("Water Level(m)")



#plt.title('title')
#plt.show()




canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand="yes")

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand="yes")

tkinter.mainloop() #창 실행

