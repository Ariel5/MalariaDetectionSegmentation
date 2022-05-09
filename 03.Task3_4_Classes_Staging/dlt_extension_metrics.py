import dlt
import dlt.common
import sklearn
import scipy

class AUC(dlt.common.monitors.ValuesCollectorNew):
    def __init__(self, pred_tag, label_tag, output_tag='var.auc', label_one_hot=False, do_train=True, do_val=True):
        super().__init__([pred_tag, label_tag], do_train, do_val)
        self.pred_tag = pred_tag
        self.label_tag = label_tag
        self.output_tag = output_tag
        self.do_train = do_train
        self.do_val = do_val
        self.label_one_hot = label_one_hot
        
    def _calculate(self, context):
        y_hat = self.value_dict[self.pred_tag]
        labels = self.value_dict[self.label_tag]

        y_hat = scipy.special.softmax(y_hat)[:,-1]
        y_hat = scipy.special.expit(y_hat)

        if self.label_one_hot:
            labels = np.argmax(labels, axis=1)
        else:
            labels = labels.ravel()

        auc = sklearn.metrics.roc_auc_score(labels, y_hat, max_fpr=None)        
        context[self.output_tag] = auc

    def callback_post_train_epoch(self, context):
        if self.do_train:
            self._calculate(context)

    def callback_post_val_epoch(self, context):
        if self.do_val:
            self._calculate(context)
