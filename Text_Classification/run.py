
# Cài đặt một số hàm tiền xử lý văn bản cần thiết
import time

import eli5 as eli5
import regex as re
from sklearn.feature_extraction.text import CountVectorizer
from underthesea import word_tokenize
 
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
 
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
dicchar = loaddicchar()

# Hàm chuyển Unicode dựng sẵn về Unicde tổ hợp (phổ biến hơn)
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)

def text_preprocess(document):
    # xóa html code
    document = remove_html(document)
    # chuẩn hóa unicode
    document = convert_unicode(document)
    # chuẩn hóa cách gõ dấu tiếng Việt
    document = chuan_hoa_dau_cau_tieng_viet(document)
    # tách từ
    document = word_tokenize(document, format="text")
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



stopword = set()
file = open('stopwords.txt', 'r',encoding='utf8')
word = file.readline()
while word:
    word = word.strip()
    # print(word)
    stopword.add(word)
    word = file.readline()
file.close()

def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
MODEL_PATH = "models"

import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

import pickle

model = pickle.load(open(os.path.join(MODEL_PATH,"svm.pkl"), 'rb'))



#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# xem kết quả cho 1 văn bản model đã load ở trên

document = """
TP HCM phạt người không đeo khẩu trang nơi công cộng
Người dân ở thành phố không đeo khẩu trang nơi công cộng sẽ bị xử phạt mức cao nhất 300.000 đồng, từ ngày 5/8.

Yêu cầu này được Chủ tịch UBND thành phố Nguyễn Thành Phong đưa ra tại cuộc họp Ban chỉ đạo phòng chống dịch bệnh Covid-19 của TP HCM chiều 3/8.

Việc xử phạt không đeo khẩu trang nơi công cộng được TP HCM cũng như các địa phương khác thực hiện từ cuối tháng 3 khi Covid-19 bùng phát. Tuy nhiên, sau khi hết thực hiện cách ly xã hội từ ngày 23/4, việc đeo khẩu trang nơi công cộng chỉ dừng lại ở mức khuyến cáo.

Theo Nghị định số 176/2013, người dân không đeo khẩu trang nơi công cộng sẽ bị xử phạt từ 100.000 đến 300.000 đồng. Trong khoảng một tháng áp dụng trước đó, TP HCM đã xử phạt hơn 4.300 trường hợp với gần 870 triệu đồng.

Theo ông Phong, việc đeo khẩu trang đã được khẳng định có thể tránh lây lan dịch bệnh cho người khác và bảo vệ sức khỏe cho người sử dụng. "Sở Công thương phải nắm nguồn cung ứng khẩu trang, chủ động thông báo các điểm bán để người dân dễ dàng mua vì đã xử phạt thì phải bảo đảm đủ nguồn cung", ông Phong nói.

Đội trật tự đô thị phường Bến Nghé, quận 1, xử phạt người không đeo khẩu trang trên phố đi bộ Nguyễn Huệ, chiều 15/4. Ảnh: Quỳnh Trần.
Đội trật tự đô thị phường Bến Nghé, quận 1, xử phạt người không đeo khẩu trang trên phố đi bộ Nguyễn Huệ, chiều 15/4. Ảnh: Quỳnh Trần.

Bí thư Thành uỷ Nguyễn Thiện Nhân cũng cho rằng việc đeo khẩu trang là một trong những biện pháp cơ bản để tránh dịch bệnh lây lan. Việc này rất dễ làm, không tốn nhiều tiền nhưng nhiều nước bỏ lơi và đã bị "vỡ trận".

"Ngoài đường hiện có ít nhất 20% người không đeo khẩu trang. Người không đeo không những tự rước bệnh vào mình mà còn nguy cơ lây cho người khác. Đeo khẩu trang hơi cực tí thôi nhưng đi đâu cũng nên đeo để giữ an toàn", ông Nhân nói và khẳng định thành phố bảo đảm không thiếu khẩu trang cho người dân.

Chủ tịch UBND thành phố Nguyễn Thành Phong cũng cho biết đã đồng ý việc tái lập các chốt kiểm soát ở cửa ngõ TP HCM để phòng chống Covid-19.

Trước đó, thành phố đã lập 62 chốt kiểm soát, hoạt động 24/24 từ ngày 4/4 để phòng chống dịch. Lực lượng tham gia là Công an thành phố, Sở Y tế, Bộ Tư lệnh thành phố, Thanh tra giao thông, Ban Quản lý An toàn thực phẩm, quản lý thị trường.

Trong đó, 16 chốt chính (cấp thành phố) đặt tại: Trạm thu phí Long Phước (cao tốc TP HCM - Long Thành - Dầu Giây), cao tốc Trung Lương, cầu Đôi (đường Trần Văn Giàu), đường Ba Làng, đường Xuyên Á (quốc lộ 22), cầu Phú Cường, cầu Vĩnh Bình, cầu vượt Sóng Thần, quốc lộ 1K, quốc lộ 50, quốc lộ 1A, cầu Đồng Nai, Bến xe Miền Tây, Bến xe miền Đông, sân bay Tân Sơn Nhất, cảng Cát Lái.

Đến ngày 23/4, chính quyền thành phố dừng hoạt động các chốt này vì dịch bệnh đã được khống chế, TP HCM dừng cách ly xã hội theo Chỉ thị 19 của Thủ tướng.

Sau 19 ngày hoạt động, các chốt chính đã kiểm tra gần 270.000 xe, trong đó có 235.000 ôtô; gần 600.000 người được kiểm tra y tế, đo thân nhiệt, bao gồm cả 3.000 người nước ngoài; hơn 130.000 người được yêu cầu khai báo y tế.
"""

document = text_preprocess(document)
document = remove_stopwords(document)

subjects = ('công nghệ','du lịch','giáo dục','giải trí','kinh doanh','nhịp sống','phim ảnh','pháp luật','sống trẻ',
            'sức khỏe','thế giới','thể thao','thời sự','thời trang','xe 360','xuất bản','âm nhạc','ẩm thực')
start_time = time.time()
label = model.predict([document])
label = int(label)
train_time = time.time() - start_time
print(train_time)
print("Accyracy: 88%")
print('Predict label:', subjects[label])
print('Showing the pickled data:')
print(model.named_steps['tfidf'])
cnt = 0
for item in model:
    print('The data ', cnt, ' is : ', item)
    cnt += 1


##form
from tkinter.font import Font
from tkinter import *
form = Tk()
endx=800 ; midx=endx/2
endy=600 ; midy=endy/2
form.geometry("800x670")
form.title('Phân loại văn bản')
form.configure(bg='#94969E')
#frame
def getTextInput():
    document=text.get("1.0","end")
    #Predict
    document = text_preprocess(document)
    document = remove_stopwords(document)
    start_time = time.time()
    label = model.predict([document])
    label = int(label)
    train_time = time.time() - start_time
    global subjects
    time_label.config(text ='Thời gian dự đoán: {}s'.format(train_time))
    print(train_time)
    result_label.config(text ='Kết quả: {}'.format(subjects[label]))
    print('Predict label:', subjects[label])

#TexArea

text = Text(form,width =95,height =28)
#scroll
scroll = Scrollbar(form, command=text.yview)
text.configure(yscrollcommand=scroll.set)
scroll.pack(side=RIGHT, fill=Y)
#font
myFont = Font(family="Times New Roman", size=12)
text.configure(font=myFont)
#~~~
text.pack()
text.place(bordermode=OUTSIDE,x=0,y=0)

#label
accur_label = Label(form,text='Độ chính xác: 88%')
accur_label.configure(font=myFont)
accur_label.pack()
accur_label.place(bordermode=OUTSIDE,x=0,y=midy+290)

time_label = Label(form,text='Thời gian dự đoán: 0s')
time_label.configure(font=myFont)
time_label.pack()
time_label.place(bordermode=OUTSIDE,x=0,y=midy+310)

result_label = Label(form,text='Kết quả: none')
result_label.configure(font=myFont)
result_label.pack()
result_label.place(bordermode=OUTSIDE,x=0,y=midy+330)


#button
button = Button(form, text='Dự đoán', width=15,command=getTextInput)
button.pack()
button.place(bordermode=OUTSIDE,x=0,y=midy+250)


form.mainloop()
