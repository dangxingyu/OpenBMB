# coding:utf-8


import torch
import bmtrain as bmt

from model_center.tokenizer import BartTokenizer
from model_center.model import BartConfig, Bart

from transformers import BartForConditionalGeneration as hugBart


def main():
    bmt.init_distributed()

    path = "/home/zhaoweilin/dxy/ModelCenter/configs/bart/bart-base/"
    # tokenizer = BartTokenizer.from_pretrained(path)
    config = BartConfig.from_pretrained(path)
    config.activation_dropout = 0
    config.dropout_p = 0
    config.attention_dropout = 0
    bmt_bart = Bart.from_pretrained(path, config=config).eval()

    # bmt_bart = torch.load('/home/zhaoweilin/smap/ModelCenter/configs/bart/bart-base/pytorch_model.pt')

    # print('load success')

    hug_bart = hugBart.from_pretrained('facebook/bart-base').cuda().eval().half()

    test_turns = 1
    for _ in range(test_turns):
        batch = 1
        max_encoder_length = 16
        max_decoder_length = 16
        input_ids = torch.randint(
            config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(
            max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        decoder_input_ids = torch.randint(
            config.vocab_size, (batch, max_decoder_length,), dtype=torch.int32).cuda()
        decoder_length = torch.randint(
            max_decoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[
            None, :].repeat(input_ids.shape[0], 1) < length[:, None]
        decoder_attention_mask = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)[
            None, :].repeat(decoder_input_ids.shape[0], 1) < decoder_length[:, None]

        bmt_logits = bmt_bart(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                              decoder_attention_mask=decoder_attention_mask, return_logits=True)
        hug_logits = hug_bart(input_ids=input_ids, attention_mask=attention_mask,
                              decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask).logits
        b = bmt_logits*decoder_attention_mask[:, :, None]
        h = hug_logits*decoder_attention_mask[:, :, None]
        print(b.shape, h.shape)
        d = (h - b).abs()
        # print(d.max(), h.abs().max(), b.abs().max())
        print(d.max())


if __name__ == "__main__":
    main()
