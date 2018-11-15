import tensorflow as tf
tf.reset_default_graph() 

import random
import numpy as np
#np.set_printoptions(threshold=np.inf)
from tensorflow.python.ops import lookup_ops
from nmt.utils import misc_utils as utils
import codecs
from tensorflow.python.layers import core as layers_core
import time
import sys
import collections
import math

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0
# res_list=[]
# one_list=[]

# def cond(i,j,n,logits):
#   return i<n

# def cond2(i,j,n,logits):
#   return j<n

# def body2(i,j,n,logits):
#   one_list.append(np.argmax(logits[i][j]))
#   j=j+1
#   return i,j,n,logits

# def body(i,j,n,logits):
#   one_list=[]
#   j=0
#   tf.while_loop(cond2,body2,[i,j,logits.shape[1],logits])
#   res_list.append(one_list)
#   i=i+1
#   return i,j,n,logits

if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  if not tf.gfile.Exists("./out/"): tf.gfile.MakeDirs("./out/")
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile("./vocabsrc", "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
      #得到词向量表
  emb_dict = dict()
  emb_size = None
  i=0
  with codecs.getreader("utf-8")(tf.gfile.GFile("./sgns.weibo.worddata", 'rb')) as f:
    for line in f:
      tokens = line.strip().split(" ")
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        assert emb_size == len(vec), "All embedding size should be same."
      else:
        emb_size = len(vec)
      i+=1
      if(i%50000 == 0):
        print(i)
  for token in ["<unk>","<s>","</s>"]:
    if token not in emb_dict:
      emb_dict[token] = [0.0] * 300
      #生成与单词表对应顺序的词向量表
  emb_matG = np.array(
  [emb_dict[token] for token in vocab], dtype=tf.float32.as_numpy_dtype())
  '''
  emb_mat_constG = tf.slice(emb_mat, [3, 0], [-1, -1])
  with tf.variable_scope("pretrain_embeddings", dtype=tf.float32) as scope:
    #with tf.device(_get_embed_device(num_trainable_tokens)):
    with tf.device("/cpu:0"):
      emb_mat_var = tf.get_variable(
      "emb_mat_var", [3, 300],
      dtype=tf.float32)
  emb_mat = tf.concat([emb_mat_var, emb_mat_const], 0)
  emb_matC = emb_mat
  '''
  #生成训练图
  graph = tf.Graph()
  with graph.as_default():
    batch_size_train = 64
    hidden_state_train = 96
    lrate = 0.001
    #加载数据,生成字符串-索引对，第二个参数为hashtable查找不到的默认返回值
    src_vocab_table = lookup_ops.index_table_from_file(
      "./vocabsrc", default_value=0)
    tgt_vocab_table = src_vocab_table
    src_dataset = tf.data.TextLineDataset("./trainsrccut.txt")
    tgt_dataset = tf.data.TextLineDataset("./traintgtcut.txt")
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
    output_buffer_size = batch_size_train * 1000
    #转化为单词表对应索引
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(EOS)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(SOS)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(EOS)), tf.int32)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    #数据处理
    src_tgt_dataset = src_tgt_dataset.skip(skip_count_placeholder)
    #src_tgt_dataset = src_tgt_dataset.shuffle(
      #output_buffer_size, 1, True)
    #第一个参数为随机抽取个数，大于dataset大小表示取整个dataset,第二个参数为random_seed,
    #第三个参数为reshuffle_echo_iterator,true表示每次调用都重新洗牌
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
          num_parallel_calls=1).prefetch(output_buffer_size)
    #string_split参数为列表，可有多个字符串项，全部分割成一个列表，values为列表值，
    #indices为每项在原列表中的索引，默认以空格分割，num_parallel_calls为并行执行个数，默认1个序列执行
    #prefetch创建后台线程开始预取数据，下次使用dataset时数据已准备好不用等待，参数为预取大小，大于dataset为取整个dataset
    src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
    #返回为true的保留，false的删除
    
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:20], tgt),
        num_parallel_calls=1).prefetch(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:20]),
        num_parallel_calls=1).prefetch(output_buffer_size)
    #截取20
    
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=1).prefetch(output_buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=1).prefetch(output_buffer_size)
    #添加开始结束字符索引，生成目标输入和输出
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=1).prefetch(output_buffer_size)
    #添加实际长度
    batched_dataset = src_tgt_dataset.padded_batch(
        batch_size_train,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused
    #对dataset做batch分组，第一个参数为batch大小，
    #第二个为padded长度，none为以最长项为基准
    #第三个为padded内容
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
      tgt_seq_len) = (batched_iter.get_next())
    #取一个batch
    with tf.device("/cpu:0"):
      initializer = tf.random_uniform_initializer(
        -0.1,0.1, seed=1)
      tf.get_variable_scope().set_initializer(initializer)
      src_ids = tf.transpose(src_ids)
      '''
      #得到单词表
      vocab = []
      with codecs.getreader("utf-8")(tf.gfile.GFile("./vocabsrc", "rb")) as f:
        vocab_size = 0
        for word in f:
          vocab_size += 1
          vocab.append(word.strip())
      #得到词向量表
      emb_dict = dict()
      emb_size = None
      i=0
      with codecs.getreader("utf-8")(tf.gfile.GFile("./sgns.weibo.worddata", 'rb')) as f:
        for line in f:
          tokens = line.strip().split(" ")
          word = tokens[0]
          vec = list(map(float, tokens[1:]))
          emb_dict[word] = vec
          if emb_size:
            assert emb_size == len(vec), "All embedding size should be same."
          else:
            emb_size = len(vec)
          i+=1
          if(i%50000 == 0):
            print(i)
      for token in ["<unk>","<s>","</s>"]:
        if token not in emb_dict:
          emb_dict[token] = [0.0] * 300
      #生成与单词表对应顺序的词向量表
      emb_mat = np.array(
        [emb_dict[token] for token in vocab], dtype=tf.float32.as_numpy_dtype())
      '''
      emb_mat = emb_matG
      emb_mat_const = tf.slice(emb_mat, [3, 0], [-1, -1])
      with tf.variable_scope("pretrain_embeddings", dtype=tf.float32) as scope:
        #with tf.device(_get_embed_device(num_trainable_tokens)):
        with tf.device("/cpu:0"):
          emb_mat_var = tf.get_variable(
          "emb_mat_var", [3, 300],
          dtype=tf.float32)
      emb_mat = tf.concat([emb_mat_var, emb_mat_const], 0)
      
      #从词向量表中查找输入词索引对应的向量，转换输入为词向量表示
      encoder_emb_inp = tf.nn.embedding_lookup(
          emb_mat, src_ids)
      #双向encodet
      cell_list = []
      #循环次数等于四层除以2(双向)
      for i in range(2):
        #生成一个lstm单元，第一个参数为单元状态个数(等于b向量个数)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
          hidden_state_train,
          forget_bias=1.0)
        #给lstm包裹dropout层
        single_cell = tf.contrib.rnn.DropoutWrapper(
          cell=single_cell, input_keep_prob=(1.0 - 0.2))
        cell_list.append(single_cell)
      fw_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
      cell_list = []
      for i in range(2):
        #生成一个lstm单元，第一个参数为单元状态个数(等于b向量个数)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
          hidden_state_train,
          forget_bias=1.0)
        #给lstm包裹dropout层
        single_cell = tf.contrib.rnn.DropoutWrapper(
          cell=single_cell, input_keep_prob=(1.0 - 0.2))
        cell_list.append(single_cell)
      bw_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
      bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        encoder_emb_inp,
        dtype=tf.float32,
        sequence_length=src_seq_len,
        time_major=True,
        swap_memory=True)
      encoder_outputs  = tf.concat(bi_outputs, -1)
      bi_encoder_state = bi_state
      encoder_state = []
      for layer_id in range(2):
        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
      encoder_state = tuple(encoder_state)
      '''
      #生成encoder四层
      cell_list = []
      for i in range(4):
        #生成一个lstm单元，第一个参数为单元状态个数(等于b向量个数)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
          hidden_state_train,
          forget_bias=1.0)
        #给lstm包裹dropout层
        single_cell = tf.contrib.rnn.DropoutWrapper(
          cell=single_cell, input_keep_prob=(1.0 - 0.2))
        cell_list.append(single_cell)
      #使用列表生成四层单列网络，动态一项项编码，列表项为各层单元(单层单元相同，不同层单元可以不同)
      mulRnnCells = tf.contrib.rnn.MultiRNNCell(cell_list)
      #生成rnn的encoder网络，动态时间序列编码
      #第二个参数为输入词向量，第一维句子长度(time major)，第二位样本数，第三位词向量，已padded到同句长
      #第三个参数为输入数据类型(词向量为浮点型)
      #第四个参数为实际句子长度
      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            mulRnnCells,
            encoder_emb_inp,
            dtype=tf.float32,
            sequence_length=src_seq_len,
            time_major=True,
            swap_memory=True)
      '''
      #生成decoder
      #支持attention
      #转成time major
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
      #两种attention机制，luong和bahdanau
      #第一个参数为神经元状态个数，第二个为encoder的每个状态输出，第三个为每个样本状态长度的列表
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        hidden_state_train, memory, memory_sequence_length=src_seq_len)
      #构造四层单列单元
      cell_list = []
      for i in range(4):
        single_cell = tf.contrib.rnn.BasicLSTMCell(
          hidden_state_train,
          forget_bias=1.0)
        single_cell = tf.contrib.rnn.DropoutWrapper(
          cell=single_cell, input_keep_prob=(1.0 - 0.2))
        cell_list.append(single_cell)
      mulRnnCells = tf.contrib.rnn.MultiRNNCell(cell_list)
      #使用attention机制，包裹attention层
      #第三个参数为神经元状态个数，第四个参数只在infer的greedy情况为Ture，
      #第五个参数为是否返回attention
      mulRnnCells = tf.contrib.seq2seq.AttentionWrapper(
        mulRnnCells,
        attention_mechanism,
        attention_layer_size=hidden_state_train,
        alignment_history=False,
        output_attention=False,
        name="attention")
      #根据encoder的输入得到decoder的初始状态
      decoder_initial_state = mulRnnCells.zero_state(batch_size_train, tf.float32).clone(
          cell_state=encoder_state)
      #生成样本输入对应正确的词向量，在训练阶段用于decoder输入
      target_input = tgt_input_ids
      target_input = tf.transpose(target_input)
      decoder_emb_inp = tf.nn.embedding_lookup(
            emb_mat, target_input)
      #traininghelper对decoder输入，生成对应helper自动处理每次输入，第三个参数为time major
      helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp,tgt_seq_len,
            time_major=True)
      '''
      with tf.variable_scope("decoder"):
        output_layer = layers_core.Dense(
           vocab_size, use_bias=False, name="output_projection")
      print(output_layer)
      '''
      #构造decoder，单列四层单元，encoder输出值为初始状态
      my_decoder = tf.contrib.seq2seq.BasicDecoder(
            mulRnnCells,
            helper,
            decoder_initial_state)
      #生成decoder网络，时间序列动态一项项解码
      #outputs为(rnn_output，sample_id)元组
      #rnn_output第一维时间序列，第二维样本数，第三维神经元状态个数(对应每个词的特征向量)
      #sample_id第一维时间，第二维样本数，对应每个词输出的选取项，一个样本对应时间序列向量的sample_id向量为样本对应预测结果
      outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=True,
            swap_memory=True,
            scope="decoder")
      
      reverse_tgt_vocab_table1 = lookup_ops.index_to_string_table_from_file(
        "/home/ubuntu/nmt/vocabsrc", default_value="<unk>")
      sample_wordssrc1 = reverse_tgt_vocab_table1.lookup(
          tf.to_int64(src_ids))
      target_input1 = reverse_tgt_vocab_table1.lookup(
          tf.to_int64(target_input))
      sample_wordstgt1 = reverse_tgt_vocab_table1.lookup(
          tf.to_int64(outputs.sample_id))
      
      #dense层将特征空间映射到样本空间，生成单词表对应的结果，
      #每个词的特征向量转换为单词表长度的向量(每个单词可能性的概率？)
      with tf.variable_scope("decoder"):
        output_layer = layers_core.Dense(
           vocab_size, use_bias=False, name="output_projection")
        logits = output_layer(outputs.rnn_output)
        logits1 = logits[:,1,:]
        #print(logits1.get_shape())
        #logitsoutput = tf.transpose(logits1,perm=[1,0,2])
        logitsoutput = logits1[:5,:]
        # logitsoutput = reverse_tgt_vocab_table1.lookup(
        #   tf.to_int64(logitsoutput))
        print(logits1.get_shape())
        #得到最大可能性单词
        #res_list=[]
        # print(logits.get_shape())
        # i=tf.Variable(0)
        # j=tf.Variable(0)
        # tf.while_loop(cond,body,[i,j,logits.shape[0],logits])
        # for i in range(logits.shape[0]):
        #   one_list=[]
        #   for j in range(logits.shape[1]):
        #     one_list.append(np.argmax(logits[i][j]))
        #   res_list[i].append(one_list)
        # res_list1 = reverse_tgt_vocab_table1.lookup(
        #   tf.to_int64(res_list))
      #print(logits)
      #正确输出处理，用于计算损失
      target_output = tgt_output_ids
      target_output = tf.transpose(target_output)
      #crossent第一维时间(padded之后的等长值)，第二维样本数，为每个词预测值和真实值之间的损失
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
      #max_time为padded后的batch内最大长度，tgt_seq_len为每句实际长度
      #target_weight为batch样本数大小的列表，每一项为max_time长，实际长度内为true，实际长度外为false的掩码
      max_time = target_output.shape[0].value or tf.shape(target_output)[0]
      target_weights = tf.sequence_mask(
        tgt_seq_len, max_time, dtype=logits.dtype)
      target_weights = tf.transpose(target_weights)
      #计算loss，每个词的损失求和为整个batch损失，再求样本平均损失
      loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(batch_size_train)
      #得到可训练变量列表，encoder和decoder的w和b，dense层的w
      params = tf.trainable_variables()
      for ele1 in params:
        print(ele1.name)
      #得到梯度，第一个参数对第二个参数求偏导
      #w和b每层encoder一样，四层有四组不同值
      #w和b每层decoder一样，四层有四组不同值
      #输出dense层w值
      #w转置*x+b，w列数和b行数相同
      gradients = tf.gradients(
          loss,
          params,
          colocate_gradients_with_ops=False)
      #裁剪梯度，第二个参数为裁剪阀值
      clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, 5.0)
      #参数为学习速率(0到1之间)
      opt = tf.train.AdamOptimizer(lrate)
      global_step = tf.Variable(0, trainable=False)
      #使用梯度按照optimizer的方式更新参数，更新后global_step自增1
      update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=global_step)
      #实例化saver，用于保存checkpoint
      saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=1)
      tf.summary.scalar('loss',loss)
      merged = tf.summary.merge_all()
      #退出graph和cpu0
  #生成session的config
  config_proto = tf.ConfigProto()
  #生成session，第二个参数为配置，第三个参数为sess用的图
  train_sess = tf.Session(
  target="", config=config_proto, graph=graph)
  with graph.as_default():
    #取最后的checkpoint，用于恢复session
    latest_ckpt = tf.train.latest_checkpoint("/home/ubuntu/nmt/ckpt/")
    if latest_ckpt:
      #使用最新checkpoint恢复session
      saver.restore(train_sess, latest_ckpt)
      train_sess.run(tf.tables_initializer())
      #global_stepInt = global_step.eval(session=train_sess)
    else:
      #不存在checkpoint则重新训练
      train_sess.run(tf.global_variables_initializer())
      train_sess.run(tf.tables_initializer())
      #global_stepInt = global_step.eval(session=train_sess)
  #writer = tf.summary.FileWriter("/tmp/tmpBoard",train_sess.graph)
  epoch_step=0
  #feed一个变量给placeholder，每次运行使用变量填充，变量值每次运行变化
  skip_count = batch_size_train * epoch_step
  train_sess.run(
  batched_iter.initializer,
  feed_dict={skip_count_placeholder: skip_count})
  #运行sess，评估global_step
  #global_stepInt = global_step.eval(session=train_sess)
  stats = {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
  "total_count": 0.0, "grad_norm": 0.0}
  info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
  "avg_grad_norm": 0.0, "learning_rate": 1.0}
  #得到当前时间
  #start_train_time = time.time()
  #保存图给tensorboard
  writer = tf.summary.FileWriter("/tmp/tmpBoard",train_sess.graph)
  #writer.close()
  #生成推理图
  graphInfer = tf.Graph()
  with graphInfer.as_default():
    hidden_state_infer = 96
    src_vocab_table2 = lookup_ops.index_table_from_file(
    "/home/ubuntu/nmt/vocabsrc", default_value=0)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        "/home/ubuntu/nmt/vocabsrc", default_value="<unk>")
    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string,name="src_placeholder")
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64,name="batch_size_placeholder")
    #'''
    #切分第一个维度，生成dataset(句子的列表)
    src_dataset2 = tf.data.Dataset.from_tensor_slices(
        src_placeholder)
    src_eos_id2 = tf.cast(src_vocab_table2.lookup(tf.constant("</s>")), tf.int32)
    #转换成词列表
    src_dataset2 = src_dataset2.map(lambda src: tf.string_split([src]).values)
    #转换成索引
    src_dataset2 = src_dataset2.map(
      lambda src: tf.cast(src_vocab_table2.lookup(src), tf.int32))
    #在后面加上长度
    src_dataset2 = src_dataset2.map(lambda src: (src, tf.size(src)))
    batched_dataset2 = src_dataset2.padded_batch(
        batch_size_placeholder,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id2,  # src
            0))  # src_len -- unused
    batched_iter2 = batched_dataset2.make_initializable_iterator()
    #batched_iter2 = batched_dataset2.make_one_shot_iterator()
    #tf.add_to_collection('bIterator', batched_iter2)
    #batched_iter2 = tf.identity(batched_iter2,name='batched_iter')
    (src_ids2, src_seq_len2) = batched_iter2.get_next()
    #'''
    src_eos_id2 = tf.cast(src_vocab_table2.lookup(tf.constant("</s>")), tf.int32)
    '''
    srcData2 = tf.string_split(src_placeholder).values
    srcData2 = tf.cast(src_vocab_table2.lookup(srcData2), tf.int32)
    (src_ids2, src_seq_len2) = (srcData2, tf.size(srcData2))
    #print("------")
    src_seq_len2 = tf.reshape(src_seq_len2,shape=[1,])
    src_ids2 = tf.reshape(src_ids2,shape=[1,tf.size(srcData2)])??
    '''
    #print(src_seq_len2)
    #print(src_ids2)
    initializer = tf.random_uniform_initializer(
    -1,1, seed=1)
    tf.get_variable_scope().set_initializer(initializer)
    src_ids2 = tf.transpose(src_ids2)
    vocab_size2 = vocab_size
    vocab2 = vocab
    '''
    #得到单词表
    vocab2 = []
    with codecs.getreader("utf-8")(tf.gfile.GFile("/home/ubuntu/nmt/vocabsrc", "rb")) as f:
      vocab_size2 = 0
      for word in f:
        vocab_size2 += 1
        vocab2.append(word.strip())
    
    #得到词向量表
    emb_dict2 = dict()
    emb_size2 = None
    i=0
    with codecs.getreader("utf-8")(tf.gfile.GFile("./sgns.weibo.worddata", 'rb')) as f:
      for line in f:
        tokens2 = line.strip().split(" ")
        word2 = tokens2[0]
        vec2 = list(map(float, tokens2[1:]))
        emb_dict2[word2] = vec2
        if emb_size2:
          assert emb_size2 == len(vec2), "All embedding size should be same."
        else:
          emb_size2 = len(vec2)
        i+=1
        if(i%50000 == 0):
          print(i)
    for token in ["<unk>","<s>","</s>"]:
      if token not in emb_dict2:
        emb_dict2[token] = [0.0] * 300
    #生成与单词表对应顺序的词向量表
    emb_mat2 = np.array(
    [emb_dict2[token] for token in vocab2], dtype=tf.float32.as_numpy_dtype())
    '''
    emb_mat2 = emb_matG
    #print(emb_mat2)
    #sqrt32 = math.sqrt(3)
    #initializerMat2 = tf.random_uniform_initializer(-sqrt3, sqrt3,seed=1)
    emb_mat_const2 = tf.slice(emb_mat2, [3, 0], [-1, -1])
    with tf.variable_scope("pretrain_embeddings", dtype=tf.float32) as scope:
        #with tf.device(_get_embed_device(num_trainable_tokens)):
      with tf.device("/cpu:0"):
        emb_mat_var2 = tf.get_variable(
        "emb_mat_var", [3, 300],
        dtype=tf.float32)
    emb_mat2 = tf.concat([emb_mat_var2, emb_mat_const2], 0)
    
    #从词向量表中查找输入词索引对应的向量，转换输入为词向量表示
    encoder_emb_inp2 = tf.nn.embedding_lookup(
          emb_mat2, src_ids2)
    #双向encodet
    cell_list2 = []
    #循环次数等于四层除以2(双向)
    for i in range(2):
      #生成一个lstm单元，第一个参数为单元状态个数(等于b向量个数)
      single_cell2 = tf.contrib.rnn.BasicLSTMCell(
        hidden_state_infer,
        forget_bias=1.0)
      #给lstm包裹dropout层
      #single_cell2 = tf.contrib.rnn.DropoutWrapper(
        #cell=single_cell2, input_keep_prob=(1.0 - 0.2))
      cell_list2.append(single_cell2)
    fw_cell2 = tf.contrib.rnn.MultiRNNCell(cell_list2)
    cell_list2 = []
    for i in range(2):
      #生成一个lstm单元，第一个参数为单元状态个数(等于b向量个数)
      single_cell2 = tf.contrib.rnn.BasicLSTMCell(
        hidden_state_infer,
        forget_bias=1.0)
      #给lstm包裹dropout层
      #single_cell2 = tf.contrib.rnn.DropoutWrapper(
        #cell=single_cell2, input_keep_prob=(1.0 - 0.2))
      cell_list2.append(single_cell2)
    bw_cell2 = tf.contrib.rnn.MultiRNNCell(cell_list2)
    bi_outputs2, bi_state2 = tf.nn.bidirectional_dynamic_rnn(
      fw_cell2,
      bw_cell2,
      encoder_emb_inp2,
      dtype=tf.float32,
      sequence_length=src_seq_len2,
      time_major=True,
      swap_memory=True)
    encoder_outputs2  = tf.concat(bi_outputs2, -1)
    bi_encoder_state2 = bi_state2
    encoder_state2 = []
    for layer_id in range(2):
      encoder_state2.append(bi_encoder_state2[0][layer_id])  # forward
      encoder_state2.append(bi_encoder_state2[1][layer_id])  # backward
    encoder_state2 = tuple(encoder_state2)
    '''
    #生成encoder四层
    cell_list2 = []
    for i in range(4):
      #生成一个lstm单元，第一个参数为单元状态个数(等于b向量个数)
      single_cell2 = tf.contrib.rnn.BasicLSTMCell(
      hidden_state_infer,forget_bias=1.0)
      cell_list2.append(single_cell2)
    #使用列表生成四层单列网络，动态一项项编码，列表项为各层单元(单层单元相同，不同层单元可以不同)
    mulRnnCells2 = tf.contrib.rnn.MultiRNNCell(cell_list2)
    #生成rnn的encoder网络，动态时间序列编码
    #第二个参数为输入词向量，第一维句子长度(time major)，第二位样本数，第三位词向量，已padded到同句长
    #第三个参数为输入数据类型(词向量为浮点型)
    #第四个参数为实际句子长度
    encoder_outputs2, encoder_state2 = tf.nn.dynamic_rnn(
    mulRnnCells2,
    encoder_emb_inp2,
    dtype=tf.float32,
    sequence_length=src_seq_len2,
    time_major=True,
    swap_memory=True)
    '''
    #生成decoder
    #支持attention
    #转成time major
    max_encoder_length2 = tf.reduce_max(src_seq_len2)
    maximum_iterations2 = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length2) * 2))
    batch_size2 = tf.size(src_seq_len2)
    memory2 = tf.transpose(encoder_outputs2, [1, 0, 2])
    '''
    memory2 = tf.contrib.seq2seq.tile_batch(
          memory2, multiplier=3)
    src_seq_len2 = tf.contrib.seq2seq.tile_batch(
          src_seq_len2, multiplier=3)
    encoder_state2 = tf.contrib.seq2seq.tile_batch(
          encoder_state2, multiplier=3)
    '''
    #batch_size2 = batch_size2 * 3
    #memory2 = encoder_outputs2
    #两种attention机制，luong和bahdanau
    #第一个参数为神经元状态个数，第二个为encoder的每个状态输出，第三个为每个样本状态长度的列表
    attention_mechanism2 = tf.contrib.seq2seq.BahdanauAttention(
        hidden_state_infer, memory2, memory_sequence_length=src_seq_len2)
    #构造四层单列单元
    cell_list2 = []
    for i in range(4):
      single_cell2 = tf.contrib.rnn.BasicLSTMCell(
      hidden_state_infer,forget_bias=1.0)
      cell_list2.append(single_cell2)
    mulRnnCells2 = tf.contrib.rnn.MultiRNNCell(cell_list2)
    #使用attention机制，包裹attention层
    #第三个参数为神经元状态个数，第四个参数只在infer的greedy情况下为True，
    #第五个参数为是否返回attention
    mulRnnCells2 = tf.contrib.seq2seq.AttentionWrapper(
      mulRnnCells2,
      attention_mechanism2,
      attention_layer_size=hidden_state_infer,
      alignment_history=True,
      output_attention=False,
      name="attention")
    #根据encoder的输入得到decoder的初始状态，
    #zero_state第一个参数为batch_size，对话系统infer只有一句话，batch大小为1
    decoder_initial_state2 = mulRnnCells2.zero_state(batch_size2, tf.float32).clone(
          cell_state=encoder_state2)
    #print(decoder_initial_state2)
    tgt_sos_id2 = tf.cast(src_vocab_table2.lookup(tf.constant("<s>")),
                         tf.int32)
    tgt_eos_id2 = tf.cast(src_vocab_table2.lookup(tf.constant("</s>")),
                         tf.int32)
    start_tokens2 = tf.fill([batch_size2], tgt_sos_id2)
    end_token2 = tgt_eos_id2
    
    #贪婪采样用于infer的输入方案
    helper2 = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                emb_mat2, start_tokens2, end_token2)
    #with tf.variable_scope("decoder"):
    output_layer2 = layers_core.Dense(
         vocab_size2, use_bias=False, name="output_projection")
    #包含dense层，输出映射到样本空间
    my_decoder2 = tf.contrib.seq2seq.BasicDecoder(
              mulRnnCells2,
              helper2,
              decoder_initial_state2,
              output_layer=output_layer2)
    '''
    output_layer2 = layers_core.Dense(
         vocab_size2, use_bias=False, name="output_projection")
    my_decoder2 = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=mulRnnCells2,
              embedding=emb_mat2,
              start_tokens=start_tokens2,
              end_token=end_token2,
              initial_state=decoder_initial_state2,
              beam_width=3,
              output_layer=output_layer2)
    '''
    outputs2, final_context_state2, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder2,
            maximum_iterations=maximum_iterations2,
            output_time_major=True,
            swap_memory=True,
            scope="decoder")
    #sample_id为词典内的词索引
    #logits2 = outputs2.rnn_output
    #logits2 = tf.identity(outputs2.rnn_output,name='outputs')
    #logits2 = tf.no_op()
    #sample_id2 = outputs2.predicted_ids
    #sample_id2 = outputs2.sample_id
    sample_id2 = tf.identity(outputs2.sample_id,name='sample_id')
    loss2 = None
    sample_words2 = reverse_tgt_vocab_table.lookup(
          tf.to_int64(sample_id2),name="sample_word")
    global_step2 = tf.Variable(0, trainable=False)
    params2 = tf.trainable_variables()
    saver2 = tf.train.Saver(tf.global_variables(), max_to_keep=1)
  infer_sess = tf.Session(
      target="", config=config_proto, graph=graphInfer)
  #得到开发集的源句和目标句
  with codecs.getreader("utf-8")(
  tf.gfile.GFile("/home/ubuntu/nmt/devsrccut.txt", mode="rb")) as f:
    sample_src_data = f.read().splitlines()
  with codecs.getreader("utf-8")(
  tf.gfile.GFile("/home/ubuntu/nmt/devtgtcut.txt", mode="rb")) as f:
    sample_tgt_data = f.read().splitlines()
  with graphInfer.as_default():
    latest_ckpt2 = tf.train.latest_checkpoint("/home/ubuntu/nmt/ckpt/")
    if latest_ckpt2:
      '''
      reader = tf.train.NewCheckpointReader("/home/ubuntu/nmt/ckpt/ckpt.ckpt-0")
      variables = reader.get_variable_to_shape_map()
      for ele in variables:  
        print(ele)
      print(latest_ckpt2)
      '''
      #使用最新checkpoint恢复session
      saver2.restore(infer_sess, latest_ckpt2)
      infer_sess.run(tf.tables_initializer())
    else:
      infer_sess.run(tf.global_variables_initializer())
      infer_sess.run(tf.tables_initializer())
    #global_stepInfer = global_step2.eval(session=infer_sess)
  #启动循环迭代
  global_stepInt = 0
  while global_stepInt < 10000000:
    start_time = time.time()
    print("start step %d:" % (global_stepInt))
    step_result = train_sess.run([loss,
    global_step,
    gradient_norm,
    outputs,
    src_ids,
    sample_wordssrc1,
    sample_wordstgt1,
    target_output,
    target_weights,
    merged,
    target_input1,
    logitsoutput,
    #decoder_initial_state,
    #target_input,
    #logits,
    #logitsI,
    #src_tgt_datasetd,
    #emb_mat,
    update])
    epoch_step += 1
    #if global_stepInt%10 == 0:
    saver.save(train_sess,"/home/ubuntu/nmt/ckpt/ckpt.ckpt",
    global_step=global_stepInt)
    (step_loss, global_stepInt,grad_norm,outputsd,src_idsd,
    sample_wordssrcd1,sample_wordstgtd1,target_outputd,target_weightsd,
    rs,target_inputd1,logitsoutput1,_) = step_result
    print((step_loss, global_stepInt,grad_norm))
    #print(params)
    print(src_idsd.transpose([1,0])[0])
    #print(outputsd.rnn_output.transpose([1,0,2])[0])
    print(outputsd.sample_id.transpose([1,0])[0])
    print(target_outputd.transpose([1,0])[0])
    print(target_weightsd.transpose()[0])
    #print(decoder_initial_stated[0])
    # print(target_inputd.transpose([1,0])[0])
    # print(logitsd.transpose([1,0,2])[0,0:3,0:10])
    # print(logitsId.transpose([1,0])[0])
    sample_wordssrcd1 = sample_wordssrcd1.transpose()
    sample_wordstgtd1 = sample_wordstgtd1.transpose()
    target_inputd1 = target_inputd1.transpose()
    # res_listd1 = res_listd1.transpose()
    # #tgt_eos = "</s>".encode("utf-8")
    # #取第一个句子结果
    output12 = sample_wordssrcd1[0, :].tolist()
    output11 = sample_wordstgtd1[0, :].tolist()
    output12 =  b" ".join(output12)
    output11 =  b" ".join(output11)
    target_inputd1 = target_inputd1[0, :].tolist()
    target_inputd1 =  b" ".join(target_inputd1)
    # reverse_tgt_vocab_table1 = lookup_ops.index_to_string_table_from_file(
    #     "/home/ubuntu/nmt/vocabsrc", default_value="<unk>")
    res_list=[]
    print(logitsoutput1.shape)
    for i in range(len(logitsoutput1)):
      res_list.append(np.argmax(logitsoutput1[i]))
    print(res_list)
    res_word=[]
    with codecs.getreader("utf-8")(tf.gfile.GFile("./vocabsrc", "rb")) as f:
      for i in range(len(res_list)):
        j=0
        f.seek(0)
        for word in f:
          # if(j==0):
          #   print(word)
          if(j==res_list[i]):
          	res_word.append(word.strip())
          	break
          j=j+1
    res_word =  " ".join(res_word)
    # res_list1 = reverse_tgt_vocab_table1.lookup(
    #   tf.to_int64(res_list))
    # res_list1 = res_list1[:].tolist()
    # res_list1 =  b" ".join(res_list1)
    # res_listd1 = res_listd1[0, :].tolist()
    # res_listd1 =  b" ".join(res_listd1)
    #logitsd= logitsd[0, :].tolist()
    
    print("train input1: %s" % (output12.decode('utf-8')))
    print("train output1: %s" % (res_word))
    #print("train output1: %s" % (output11.decode('utf-8')))
    print("train ori output: %s" % target_inputd1.decode('utf-8'))
    print("logitsoutput len : %d" % (len(logitsoutput1)))
    #rs=train_sess.run(merged)
    writer.add_summary(rs, global_stepInt)
    #推理输出
    decode_id = random.randint(0, len(sample_src_data) - 1)
    print("input %s" % (sample_src_data[decode_id]))
    if not sample_src_data[decode_id]:
      continue
    iterator_feed_dict2 = {
      src_placeholder: [sample_src_data[decode_id]],
      batch_size_placeholder: 1,
    }
    infer_sess.run(batched_iter2.initializer, feed_dict=iterator_feed_dict2)
    with graphInfer.as_default():
      latest_ckpt2 = tf.train.latest_checkpoint("/home/ubuntu/nmt/ckpt/")
      print(latest_ckpt2)
      if latest_ckpt2:
        #使用最新checkpoint恢复session
        saver2.restore(infer_sess, latest_ckpt2)
        #infer_sess.run(tf.tables_initializer())
      else:
        infer_sess.run(tf.global_variables_initializer())
        infer_sess.run(tf.tables_initializer())
      #global_stepInfer = global_step2.eval(session=infer_sess)
    #得到推理词列表结果
    sample_ids_infer,sample_words_infer,src_idsd2, src_seq_lend2,encoder_emb_inpd2,outputsd2 = infer_sess.run([sample_id2, sample_words2,
    src_ids2, src_seq_len2,encoder_emb_inp2,outputs2],feed_dict=iterator_feed_dict2)
    #print(src_idsd2, src_seq_lend2,encoder_emb_inpd2,outputsd2.sample_id2)
    #结果为time_major结果，转置回一行一个句子
    #sample_ids_infer = sample_ids_infer.transpose()
    #sample_words_infer = sample_words_infer.transpose()
    sample_ids_infer = sample_ids_infer.transpose([1,0])
    sample_words_infer = sample_words_infer.transpose([1,0])
    tgt_eos = "</s>".encode("utf-8")
    #取第一个句子结果
    outputids = sample_ids_infer[0, :].tolist()
    output = sample_words_infer[0, :].tolist()
    #print("outputids: %s" % (outputids))
    #print("output: %s" % (output))
    #截取到结束符
    if tgt_eos and tgt_eos in output:
      output = output[:output.index(tgt_eos)]
    if (not hasattr(output, "__len__") and  # for numpy array
      not isinstance(output, collections.Iterable)):
      output = [output]
    translation = b" ".join(output)
    print("nmt: " + translation.decode('utf-8'))
    #saver2.save(infer_sess,"/home/ubuntu/nmt/ckptInfer/ckpt.ckpt",
    #global_step=0)
    print(global_stepInt)
    print(epoch_step)
    #if global_stepInt%18125 == 0: 
    if global_stepInt%100== 0:
      epoch_step = 0
      train_sess.run(
      batched_iter.initializer,
      feed_dict={skip_count_placeholder: skip_count})
      #with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.tables_initializer())
        #sess.run(batched_iter.initializer,feed_dict={skip_count_placeholder: 0})
        #writer = tf.summary.FileWriter("/tmp/tmpBoard",sess.graph)
        #print("00000000")
      #writer.close()
      #sys.exit()
    #encoder_emb_inp = tf.nn.embedding_lookup(
     #     self.embedding_encoder, source)
