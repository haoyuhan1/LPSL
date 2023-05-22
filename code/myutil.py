import sys
import datetime

def sort_trials(trials, key=None):
    if key is None:
        return trials
    # import ipdb; ipdb.set_trace()
    if not key in trials[0].params.keys():
        return trials
        
    def get_key(trial):
        return trial.params[key]
    trials.sort(key=get_key)
    return trials

def set_signal_by_label(x, data):
    mask = data.train_mask
    y = data.y
    x[:,:] = 0
    x[mask, data.y[mask]] = 1
    return x

class Tee:
    def __init__(self, args):
        self.args = args
        self.file = open(self.generate_file_name(), "w")
        self.stdout = sys.stdout
        
    def generate_file_name(self):
        time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        if self.args.fix_num > 0:
            num = str(self.args.fix_num)
        else:
            num = 'fix'
        if self.args.LP is True:
            model = self.args.model + '_LP'
        else:
            model = self.args.model
        filename = '../logs/' + self.args.dataset + '_' + num + '_' + model + '_' + time_stamp + '.log'
        return filename

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()