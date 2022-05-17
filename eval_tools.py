import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import random
import torch

import utils


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 주기적인 간격에 이 locator가 tick을 설정
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluateRandomly(encoder,
                     decoder,
                     pairs,
                     max_length,
                     input_lang,
                     output_lang,
                     device,
                     SOS_token,
                     EOS_token,
                     n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder=encoder,
                                            decoder=decoder,
                                            sentence=pair[0],
                                            max_length=max_length,
                                            input_lang=input_lang,
                                            output_lang=output_lang,
                                            device=device,
                                            SOS_token=SOS_token,
                                            EOS_token=EOS_token
                                            )
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate(encoder,
             decoder,
             sentence,
             max_length,
             input_lang,
             output_lang,
             device,
             SOS_token,
             EOS_token):
    with torch.no_grad():
        input_tensor = utils.tensorFromSentence(lang = input_lang,
                                                sentence=sentence,
                                                EOS_token=EOS_token,
                                                device=device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]