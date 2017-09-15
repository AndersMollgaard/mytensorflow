import matplotlib.pyplot as plt
import numpy as np
import pickle

#############################################################

class Present():
    
    def __init__(self, model_dirs, x='history', y='auc', condition={'future': 4},
                 group=['channel']):
        self.x = x
        self.y = y
        self.condition = condition
        self.group = group
        # create a list of results - each element being a dictionary of values
        results = []
        for model_dir in model_dirs:
            # the results list can hold values from several model_dirs
            with open('%s/metrics.p' %model_dir, 'rb') as f:
                metrics = pickle.load(f)
                # we add the model name for each result element
                model_name = model_dir.split('/')[-1]
                for res in metrics:
                    res['model'] = model_name
                results += metrics
        self.results = results
    
    def _group_string(self, res):
        # create a string from the concatenation of values defined in the group
        s = '#'.join('%s=%s' %(key, str(res[key])) for key in self.group if key in res)
        return s
    
    def filter_and_group(self):
        # filter results according to condition
        results_filtered = [res for res in self.results if not False in 
                            [res[key] == val for key, val in self.condition.items()]]
        # find the posible value combinations to group by
        group_strings = set( self._group_string(res) for res in results_filtered)
        # create a dictionary with results for each group
        results_dic = {group_string: {'x': [], 'y': []} for group_string in group_strings}
        for res in results_filtered:
            res_string = self._group_string(res)
            results_dic[res_string]['x'].append(res[self.x])
            results_dic[res_string]['y'].append(res[self.y])
        # sort the results of the group according to x
        for res_string, res_group in results_dic.items():
            x = np.array(res_group['x'])
            y = np.array(res_group['y'])
            sort_indices = np.argsort(x)
            res_group['x'] = x[sort_indices]
            res_group['y'] = y[sort_indices]
        return results_dic
        
    def plot(self, xlim=(), ylim=(), legend_loc=0):
        # get results_dic
        results_dic = self.filter_and_group()
        # create figure and axes
        plt.figure(figsize=(10,8))
        ax = plt.axes([0.14, 0.14, 0.79, 0.79])
        # plot each res_group
        for res_string in sorted(results_dic.keys()):
            res_group = results_dic[res_string]
            x, y = res_group['x'], res_group['y']
            plt.plot(x, y, linewidth=5, label=res_string)
        # set axes names
        plt.xlabel(self.x, fontsize=25)
        plt.ylabel(self.y, fontsize=25)
        # set axes ticks
        ax.tick_params(axis='both', which='major', labelsize=20,pad=8,length=5,width=2)
        # set axes linewidth
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        # set range of axes
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        # create legend
        leg = plt.legend(loc=legend_loc,fontsize=15)
        leg.get_frame().set_linewidth(2)
        # create title
        title = '#'.join('%s=%s' %(key, str(val)) for key, val in self.condition.items())
        plt.title(title, fontsize=30, y=1.01)
        
        