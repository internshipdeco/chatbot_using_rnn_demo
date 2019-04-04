'''
Created by Shrikrishna Yadav 
refered from UDEMY lect : "BUILDING CHATBOT USING DL"
For further improverment or any guidence your are welcome 
'''


import numpy as np
import re
import time
import tensorflow as tf

lines = open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conversations = open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

#print((conversation))
#print("hello")

#creating a dictionary that maps each lines and its id

id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
#print(id2line)


#creating a list of all of the conversation

conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(","))

print(conversations_ids[0])

#getting seperately question and answer

questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

#print(questions)
#print(answers)
#print("hello")


#first cleaning of text
def clear_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am",text)
    text = re.sub(r"he's", "he is ",text)
    text = re.sub(r"she's", "she is ",text)
    text = re.sub(r"that's", "that is ",text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"'ll", "will", text)
    text = re.sub(r"'ve", "have", text)
    text = re.sub(r"'re", "are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub('[^A-Za-z0-9 ]', "", text)
    return text

#cleaning the question
clear_question = []
for question in questions:
    clear_question.append(clear_text(question))

#cleaning the answer
clear_answer = []
for answer in answers:
    clear_answer.append(clear_text(question))

#print(clear_question[0])
#print(clear_answer[0])

#print each words with it occurence
word2count = {}
for question in clear_question:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


for answer in clear_answer:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1



#print(word2count)

#create seperate dictioinary for question and answer occurence

threshod = 10
questionwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count > threshod:
        questionwords2int[word] = word_number
        word_number += 1

answerwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count > threshod:
        answerwords2int[word] = word_number
        word_number += 1


#print(questionwords2int)
#print(answerwords2int)


#adding last token to these 2 dicitonary

tokens = ["<PAD>","<EOS>","<OUT>","<SOS>"]
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1

for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1


# creating the inverse dict of the answerword2int dict

answerint2words = {w_i:w for w, w_i in answerwords2int.items()}



#add end of string over every answer

for i in range (len(clear_answer)):
    clear_answer[i] += " <EOS>"

#print(clear_answer)

#translate all the question and the answer into integers
#and replaceing all the words that were filtered out bt  <OUT>

questions_to_int = []
for question in clear_question:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    questions_to_int.append(ints)


answers_to_int = []
for answer in clear_answer:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers_to_int.append(ints)

#print(questions_to_int)
#print(answers_to_int)

#sorting questions and answer by the length of questions

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])

#print(sorted_clean_answer)
#for quet in sorted_clean_questions:
#    if len(quet) == 4:
#        print(quet)
#print(len(sorted_clean_questions))

###  BUILDING THE SEQ2SEQ MODEL ###


#create a placeholder for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = "input")
    targets = tf.placeholder(tf.int32, [None,None], name = "target")
    learning_rate = tf.placeholder(tf.float32, name = "learning_rate")
    keep_prod = tf.placeholder(tf.float32, name = "keep_prob")
    return inputs, targets, learning_rate, keep_prod

#preprocessing the input

def preprocess_targets(targets, word2int, batch_size):

    left_side = tf.fill([batch_size,1], word2int["<SOS>"])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side], 1)
    return preprocessed_targets

#Creating the Encoder RNN Layer

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    #for below line given function return two arguments bt we dnt need 1st i.e encoder_output hence we can replace that with _
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell,
                                                       sequence_length = sequence_length, inputs = rnn_inputs,
                                                       dtype = tf.float32)

    return encoder_state

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")

    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                      training_decoder_function,
                                                                                                      decoder_embedded_input,
                                                                                                      sequence_length,
                                                                                                      scope = decoding_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)

    return output_function(decoder_output_dropout)


#Decoding the test/validation set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximun_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                                  encoder_state[0],
                                                                                  attention_keys,
                                                                                  attention_values,
                                                                                  attention_score_function,
                                                                                  attention_construct_function,
                                                                                  decoder_embeddings_matrix,
                                                                                  sos_id,
                                                                                  eos_id,
                                                                                  maximun_length,
                                                                                  num_words,
                                                                                  name = "attn_dec_inf")

    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                      test_decoder_function,
                                                                                                      scope = decoding_scope)



    return test_predictions

#creating decoder RNN

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, bacth_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                     num_words,
                                                                     None,
                                                                     scope = decoding_scope,
                                                                     weights_initializer = weights,
                                                                     biases_initializer = biases)
        training_predicitons = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   bacth_size)
        decoding_scope.reuse_variables()
        test_predections = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int["<SOS>"],
                                           word2int["<EOS>"],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           bacth_size)

    return training_predicitons, test_predections

#building seq2seq model

def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answer_num_words, question_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionwords2int):
    encoder_embededd_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answer_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embededd_input,
                                rnn_size,
                                num_layers,
                                keep_prob,
                                sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionwords2int, batch_size)
    decoder_embedding_matrix = tf.Variable(tf.random_uniform([question_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedding_input = tf.nn.embedding_lookup(decoder_embedding_matrix, preprocessed_targets)
    training_predicitions, test_predections = decoder_rnn(decoder_embedding_input,
                                                          decoder_embedding_matrix,
                                                          encoder_state,
                                                          question_num_words,
                                                          sequence_length,
                                                          rnn_size,
                                                          num_layers,
                                                          questionwords2int,
                                                          keep_prob,
                                                          batch_size)

    return training_predicitions, test_predections

#Training the seq2seq model

#setting the hyperparamters

epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#defining a session

tf.reset_default_graph()
session = tf.InteractiveSession()

#loading the model input

inputs, targets, lr, keep_prob = model_inputs()

#setting the sequence length

sequence_length = tf.placeholder_with_default(25, None, name = "sequence_length")

# getting the shape of the inputs tensor

input_shape = tf.shape(inputs)

# getting the training and test predections

training_predictions, test_predections = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerwords2int),
                                                       len(questionwords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionwords2int)

#setting up the Loss Error , the Optimizer , the gradient Clipping

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradient = optimizer.compute_gradients(loss_error)
    clipped_gradient = [ (tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradient if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradient)

#Padding the sequence with the <PAD> token
#question :[ 'WHO', 'ARE', 'YOU', <PAD>, <PAD>, <PAD>,<PAD> ]
#answer : [  <SOS>, "i ", "AM", "BOT" ,<EOS>,<PAD>,<PAD>]

def apply_padding (batch_of_sequences, word2int):
    max_sequences_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int["<PAD>"]] * (max_sequences_length - len(sequence))for sequence in batch_of_sequences]

#spilliting the data into batches of questions and answers

def split_into_batches(questions, answers, batch_size):
    for batch_index in range (0,len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        paddes_questions_in_batch = np.array(apply_padding(questions_in_batch, questionwords2int))
        paddes_answers_in_batch = np.array(apply_padding(answers_in_batch, answerwords2int))

        yield paddes_questions_in_batch, paddes_answers_in_batch

#splitting the question answer in training and validation set

trainind_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[trainind_validation_split: ]
training_answers = sorted_clean_answers[trainind_validation_split: ]
validation_question = sorted_clean_questions[:trainind_validation_split]
validation_answers = sorted_clean_answers[:trainind_validation_split]


# Training

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) -1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbit_weight.ckpt"
session.run(tf.global_variables_initializer())

for epoch in range (1, epochs+1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions,training_answers,batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs : padded_questions_in_batch,
                                                                                              targets : padded_answers_in_batch,
                                                                                              lr : learning_rate,
                                                                                              sequence_length: padded_answers_in_batch.shape[1],
                                                                                              keep_prob : keep_probability})

        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print("epoch : {:>3}/{}, Batch: {:>4}/{}, Training Loss Error : {:>6.3f}, Training Time on 100 Batchs: {:d} seconds". format(epoch,
                                                                                                                                         epochs,
                                                                                                                                         batch_index,
                                                                                                                                         len(training_questions) // batch_size,
                                                                                                                                         total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                         int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0

        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_question, validation_answers, batch_size)):

                batch_validation_loss_error = session.run(loss_error,
                                                           {inputs: padded_questions_in_batch,
                                                            targets: padded_answers_in_batch,
                                                            lr: learning_rate,
                                                            sequence_length: padded_answers_in_batch.shape[1],
                                                            keep_prob: 1})

                total_validation_loss_error += batch_validation_loss_error

            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error/ (len(validation_question) / batch_size)
            print ("validation loss error : {:>6.3f}, Batch Validation Time : {:d} seconds ". format(average_validation_loss_error,
                                                                                                         int(batch_time)))

            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min (list_validation_loss_error):
                print("I SPEAK BETTER NOW!")
                early_stopping_check = 0
                saver =  tf.train.Saver()
                saver.save((session, checkpoint))

            else:
                 print("i do not speak better , i need to practice more !")
                 early_stopping_check += 1
                 if early_stopping_check == early_stopping_stop:
                    break

    if early_stopping_check == early_stopping_stop:
        print("My apologies , I cant speak Better anymore, this the best i can do ")
        break
print("GAME OVER")

#testing the weights and running the sessions

checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

#converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clear_text(question)
    return [word2int.get(word, word2int["<OUT>"]) for word in question.split()]

#setting up the chat

while(True):
    question = input("YOU : ")
    if question == "Goodbye":
        break
    question = convert_string2int(question, questionwords2int)
    question = question + [questionwords2int["<PAD>"]] * (20 - len(question))
    fake_batch = np.zeros(batch_size, 20)
    predicated_answer = session.run(test_predections, {inputs : fake_batch, keep_prob : 0.5})[0]
    answer = ''
    for i in np.argmax(predicated_answer, 1):
        if answerint2words[i] == 'i':
            token = "I"
        elif answerint2words[i] == "<EOS>":
            taken = " . "
        elif answerint2words[i] == "<OUT>":
            taken = " OUT "
        else:
            token = " " + answerint2words[i]

        answer += token

        if token == ".":
            break

    print ("CHATBOT : " + answer)
