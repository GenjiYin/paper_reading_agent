"""
需要优化论文的分块机制！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
"""

import torch
import gradio as gr
from RAGtools import tools
from threading import Thread
from langchain.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoModelForSeq2SeqLM

project_name = "时序生成"
model_dir = r'C:\Users\96541\Desktop\千问模型大全\14Bawq量化'
translator_path = r"C:\Users\96541\Desktop\huggingface_translation_model-master\opus-mt-zh-en"
embedding_model_dir = r"C:\Users\96541\Desktop\千问模型大全\m3e"
pdf_path = './static/时序生成.pdf'
language = 'en'


history_call = [
    {"role": "你是一个金融研究专家"}
]

def init_model():
    global tokenizer, model, streamer, translator_tokenizer, translator_model, RAGtool

    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['.', '。'],
        chunk_size = 550,
        chunk_overlap  = 10,
        length_function = len,
        add_start_index = True,
    )
    sentence = []
    for p in pages:
        sentence += text_splitter.split_text(p.page_content)

    embedding_model = SentenceTransformer(embedding_model_dir)
    embeddings = embedding_model.encode(sentence)

    RAGtool = tools(name=project_name, model=embedding_model)
    RAGtool.save(sentence, embeddings)

    if language == 'en':
        translator_tokenizer = AutoTokenizer.from_pretrained(translator_path, trust_remote_code=True)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(translator_path)
    else:
        translator_tokenizer = None
        translator_model = None

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map='cuda:0', 
        trust_remote_code=True
    ).eval()
    
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )

def translater(query, model, tokenizer):     # 对提示词进行翻译
    if isinstance(query, str):
        query = [query]
    
    data = tokenizer.prepare_seq2seq_batch(src_texts=query)
    data['input_ids'] = torch.Tensor(data['input_ids']).long()
    data['attention_mask'] = torch.Tensor(data['attention_mask'])
    translation = model.generate(**data)
    out = tokenizer.batch_decode(translation, skip_special_tokens=True)
    return out

with gr.Blocks() as demo:
    init_model()

    chatbot = gr.Chatbot(
        height=700, 
        # avatar_images=()
    )

    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def clear_history():
        global history_call

        history_call = [
            {"role": "作为一个AI领域的学术专家, 你有义务为我解答任何关于论文中的问题。"}
        ]
    
    def respond(message, history):
        global history_call, tokenizer, model, streamer

        if language == 'en':
            en_prompt = translater(query=message, model=translator_model, tokenizer=translator_tokenizer)[0]
        else:
            en_prompt = message
        extra_information = RAGtool.query(en_prompt, topk=3)

        rag_prompt = """
        可供参考的资料如下: 
        """
        for i in range(len(extra_information)):
            rag_prompt += '\n 资料{}: {}'.format(i+1, extra_information[i].replace('\n', ''))
        
        rag_prompt += '\n请尽可能根据我提供的资料回答我的问题:\n'
        rag_prompt += message

        history_call.append({"role": "user", "content": rag_prompt})
        history_str = tokenizer.apply_chat_template(
            history_call,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(history_str, return_tensors='pt').to('cuda:0')

        print(rag_prompt)

        history.append([message, ""])

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
            num_beams=1,
            do_sample=True,
            top_p=0.8,
            temperature=0.3
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            history[-1][1] += new_text
            yield "", history
        
        history_call.append(
            {"role": "assistant", "content": history[-1][1]}
        )
    
    clear.click(clear_history)
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == '__main__':
    demo.launch()