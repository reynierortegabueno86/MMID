import base64
from PIL import Image
import sqlite3
import hashlib
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel,\
    BertConfig, BertweetTokenizer,  DebertaTokenizer, DebertaModel
modelByLanguage={}
modelByLanguage["en"] = {"roberta": "roberta-base", "bert": "bert-base-uncased", 'bertweet': 'vinai/bertweet-base',
                         'deberta': 'microsoft/deberta-base'}

def getModelTokenizerConfig(modelName, lang):
    config, model, tokenizer = None, None, None
    if lang in modelByLanguage:
        if modelName in ["deberta"]:
            tokenizer = DebertaTokenizer.from_pretrained(modelByLanguage[lang][modelName] )
            model = DebertaModel.from_pretrained(modelByLanguage[lang][modelName])         
        elif modelName in modelByLanguage[lang]:
            model = AutoModel.from_pretrained(
                modelByLanguage[lang][modelName])
            if modelName == 'bertweet':
                tokenizer = BertweetTokenizer.from_pretrained(modelByLanguage[lang][modelName])
            else:
                tokenizer = AutoTokenizer.from_pretrained(modelByLanguage[lang][modelName])
    return model, tokenizer

class SqliteDB:
    def __init__(self, dbname='database/urls.sqlite3'):
        self.db = sqlite3.connect(dbname)
        self.db.execute('''CREATE TABLE IF NOT EXISTS urls
              (hash BLOB PRIMARY KEY, url TEXT)''')

    def shorten(self, url):
        h = sqlite3.Binary(hashlib.sha256(url.encode('ascii')).digest())
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO urls VALUES (?, ?)', (h, url))
        return base64.urlsafe_b64encode(h).decode('ascii')

    def geturl(self, shortened_url):
        h = sqlite3.Binary(base64.urlsafe_b64decode(shortened_url.encode('ascii')))
        with self.db:
            url = self.db.execute('SELECT url FROM urls WHERE hash=?', (h,)).fetchone()
        if url is None:
            raise KeyError(shortened_url)
        return url[0]

    def close(self):
        if self.db != None:
            self.db.close()

def replace_param_value(params, pvalues):
    info=dict(params)
    for key in pvalues:
        if key in info: info[key]=pvalues[key](info[key])
    return list(info.items())

class BertConfig:
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

def getNameFileFromPath(inp, sep=os.sep):
    return inp.rsplit(sep, 1)[-1].rsplit(".", 1)[0]

def loadImage(path):
    ima = None
    try:
        ima = Image.open(str(path))
    except:
        ima = Image.new("RGB", (254, 254), color=0)
        print(f"The image {path} can't be loaded")
    ima = ima.convert("RGB")
    return ima

def loadAllImages(pathFolder):
    images={}
    print('images_dir (absolute) = ' + os.path.abspath(pathFolder))
    for root, subdirs, files in os.walk(pathFolder):
        for filename in files:
            file_path = os.path.join(root, filename)
            print('\t- file %s (full path: %s)' % (filename, file_path))
            if filename.endswith(".jpg"):
                imgname=filename.rsplit(".",1)[0]
                images[imgname]=loadImage(file_path)
    return images

