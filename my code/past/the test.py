
from fastNLP.embeddings import BertEmbedding
from fastNLP.models import BertForQuestionAnswering
from fastNLP.core.losses import CMRC2018Loss
from fastNLP.core.metrics import CMRC2018Metric
from fastNLP.io.pipe.qa import CMRC2018BertPipe
from fastNLP import Trainer, BucketSampler
from fastNLP import WarmupCallback, GradientClipCallback
from fastNLP.core.optimizer import AdamW


data_bundle = CMRC2018BertPipe().process_from_file()
data_bundle.rename_field('chars', 'words')

print(data_bundle)

embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='cn', requires_grad=True, include_cls_sep=False, auto_truncate=True,
                      dropout=0.5, word_dropout=0.01)
model = BertForQuestionAnswering(embed)
loss = CMRC2018Loss()
metric = CMRC2018Metric()

wm_callback = WarmupCallback(schedule='linear')
gc_callback = GradientClipCallback(clip_value=1, clip_type='norm')
callbacks = [wm_callback, gc_callback]

optimizer = AdamW(model.parameters(), lr=5e-5)

trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer,
                  sampler=BucketSampler(seq_len_field_name='context_len'),
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric,
                  callbacks=callbacks, device=0, batch_size=6, num_workers=2, n_epochs=2, print_every=1,
                  test_use_tqdm=False, update_every=10)
trainer.train(load_best_model=False)