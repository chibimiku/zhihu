#对data做处理方便后续加载.
import data_utils
import jieba

#用jieba对中文语料进行分词并分割
#在分割的同时提取所有的词输出词表
def gen_cut_file_jieba(input_file, cut_outputfile, vocabulary_outfile, vec_outputfile, start_header = [], appendword_file = ''):
    '''
    input_file: 输入原始文件
    cut_outputfile: 分词后的原文文件
    vocabulary_outfile: 分词后产生的词表文件
    start_header: list， 支持在词表头强行加入一系列数据，用于标识如__UNK___
    vec_outputfile: 分词后基于上述产生词表的 token_id 文件.利用data_utils里的函数.实际训练时加载vec_outputfile来生成内存里的对照表vocab
    # 上面那个功能是data.utils(initialize_vocabulary)
    '''
    print ("going to cut file...")
    vocabulary = []
    outfileobj = open(cut_outputfile, 'w+', encoding = 'utf-8')
    #先输入头部的几个tag
    for word in start_header:
        vocabulary.append(word)
    print ("goting to read input file " + input_file)
    #读取原始文件
    progress_line = 0 
    with open(input_file, "r", encoding = "utf8") as f:
        for line in f:
            if(progress_line % 2500 == 0):
                print ("proc for " + str(progress_line) + " line(s)...")
                print ("vocabulary size: " + str(len(vocabulary)))
            seg_list = jieba.lcut(line.strip(), cut_all=False) #lcut直接拿list
            #输出文件
            outfileobj.write(" ".join(seg_list))
            outfileobj.write("\n")
            #维护词表
            for single_seg in seg_list:
                if(not single_seg in vocabulary):
                    vocabulary.append(single_seg)
            progress_line = progress_line + 1
    #通过appendword_file追加3500个常用汉字.
    if (not appendword_file == "" ):
        print ("going to open appendfile:" + str(appendword_file))
        with open(appendword_file, 'r', encoding = "utf8") as apf:
            for line in apf:
                vocabulary.append(line.strip())
    print ("output vocabulary to file:" + vocabulary_outfile)
    #输出词表
    vocabulary_fileobj = open(vocabulary_outfile, 'w+', encoding = 'utf-8')
    for word in vocabulary:
        vocabulary_fileobj.write(word)
        vocabulary_fileobj.write("\n")
        
    #处理之后，根据上一步处理生成的分割结果和词典，转化一份实际的id list用于计算.
    #def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None, normalize_digits=True):
    data_utils.data_to_token_ids(cut_outputfile, vec_outputfile, vocabulary_outfile)
    
#获取词典.这里稍微封装了一下，支持把反向单词列表如['dog','cat']给dict化
#注意给出的都是byte（tf.compat.as_bytes()的产物），不是string.

'''
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
'''
def get_vocab_from_file(vocabulary_path, rev_to_dict = False):
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)
    rev_vocab = convet_list_value_from_byte_to_str(rev_vocab)
    vocab = convet_dict_value_from_byte_to_str(vocab)
    rev_dict_vocab = dict(enumerate(rev_vocab))
    return (vocab, rev_vocab, rev_dict_vocab)

def convet_list_value_from_byte_to_str(in_list, encoding="utf8"):
    new_list = []
    for v in in_list:
        new_list.append(v.decode(encoding))
    return new_list
    
def convet_dict_value_from_byte_to_str(in_dict, encoding="utf8"):
    new_dict = {}
    for k,v in in_dict.items():
        if(isinstance(k, bytes)):
            new_dict[k.decode(encoding)] = v
        else:
            new_dict[k] = v
    return new_dict
    
def get_wordlist_from_vocab_text(content_path, rev_dict_vocab):
    print ("got first word as:" + str(rev_dict_vocab[0]))
    with open(content_path, 'r', encoding="utf8") as f:
        text=f.read()
    return data_utils.sentence_to_token_ids(text, rev_dict_vocab)
 
def test():
    gen_cut_file_jieba('data/galactic_heroes.txt', 'data/galactic_heroes_cut.txt', 'data/galactic_heroes_vocab.txt', 'data/galactic_heroes_vec.txt', start_header = ['__NEWLINE__'])
    #vocab, rev_vocab = get_vocab_from_file('data/test_vocab.txt')
    #print (vocab)
    #print (rev_vocab)

if __name__=='__main__':
    test()