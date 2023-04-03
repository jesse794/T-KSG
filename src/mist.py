import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
import MLP
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

#   removes an element from a set
def remove_element(set, element):
    new_set = [set[i] for i in range(len(set)) if set[i] != element]
    return new_set

def box_counting(data, num_scales, num_bins):
    """computing the fractal dimension
       considering only scales in a logarithmic list"""
    data = pl.array(data)
    scales=np.logspace(0.01, 1, num=num_scales, endpoint=False, base=2)
    Ns=[]
    # looping over several scales
    for scale in scales:
        print ("======= Scale :",scale)
        # computing the histogram
        #bins = [np.arange(0,len(data[0]),scale) for i in range(len(data[0]))]
        bins = [num_bins for i in range(len(data[0]))]
        print(bins)
        H, edges = np.histogramdd(data, bins=(bins))
        Ns.append(np.sum(H>0))
        # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)
    return coeffs

class Node:

    def __init__(self,val,parent,state,states_left):
        self.val = val
        self.parent = parent
        self.state = state
        self.states_left = states_left
        self.lchild = None
        self.rchild = None
        self.avg_mi = 0
        self.num_wins = 0
        self.num_visits = 0
        self.lchild_visits = 0
        self.rchild_visits = 0
        self.end_state = False
        self.end_state_mi = 0
        self.visits = 0
        self.lchild_switch = False
        self.rchild_switch = False

    def eval(self):
        """evaluation function for a node, essentially just the win ratio + multi-armed bandit policy"""
        global total_visits
        global total_playouts
        #print(self.val,self.num_wins,total_visits,self.num_visits,np.sqrt(2*np.log(total_visits)/(self.num_visits+1)))
        #return (self.num_wins/(self.num_visits+1) + np.sqrt(2*np.log(self.parent.num_visits+1)/(self.num_visits+1)))
        return (self.avg_mi/(self.visits+1) + np.sqrt(2*np.log(self.parent.num_visits+1)/(self.num_visits+1)))
        #return (self.num_wins/(self.num_visits+1) + np.sqrt(2*np.log(total_visits)/(self.num_visits+1)))
        #return (self.avg_mi - self.num_visits/(self.parent.lchild_visits+self.parent.rchild_visits))

    def choose_child(self):
        """choose a child node based on the tree policy"""
        global total_visits
        self.num_visits += 1
        total_visits += 1
        if(self.val != 0):
            if(self.val > 0):
                self.parent.lchild_visits += 1
            else:
                self.parent.rchild_visits += 1
        if (self.lchild_switch == True):
            return self.rchild
        if (self.rchild_switch == True):
            return self.lchild
        #   if right and left children are empty
        #   choose a random child from states_left
        if (self.lchild == None and self.rchild == None):
            #   choose random child
            if(np.random.randint(0,2,1)[0] == 0):
                #   left child
                #random.shuffle(self.states_left)
                choice = self.states_left[0]
                #   create left child and return it
                new_states_left = remove_element(self.states_left,choice)
                new_states = self.state + [choice]
                self.lchild = Node(choice,self,new_states,new_states_left)
                self.rchild = Node(-choice,self,self.state,new_states_left)
                return self.lchild
            else:
                #   right child
                #random.shuffle(self.states_left)
                choice = self.states_left[0]
                #   create right child and return it
                new_states_left = remove_element(self.states_left,choice)
                new_states = self.state
                self.lchild = Node(choice,self,new_states+[choice],new_states_left)
                self.rchild = Node(-choice,self,new_states,new_states_left)
                return self.rchild
        #   else if lchild has not been visited but right child has
        elif (self.lchild == None and self.rchild != None):
            #   left child
            #random.shuffle(self.states_left)
            choice = self.states_left[0]
            #   create left child and return it
            new_states_left = remove_element(self.states_left,choice)
            new_states = self.state + [choice]
            self.lchild = Node(choice,self,new_states,new_states_left)
            return self.lchild
        #   else if rchild has not been visited but left child has
        elif (self.rchild == None and self.lchild != None):
            #   left child
            #random.shuffle(self.states_left)
            choice = self.states_left[0]
            #   create left child and return it
            new_states_left = remove_element(self.states_left,choice)
            new_states = self.state
            self.rchild = Node(-choice,self,new_states,new_states_left)
            return self.rchild
        #   unless both have been visited
        else:
            #   check value
            if (self.lchild.num_visits > 100 and self.rchild.num_visits > 100):
                if(self.lchild.avg_mi/(self.lchild.visits+1) < (self.rchild.avg_mi/((self.rchild.visits+1)))):
                    self.lchild_switch = True
                if(self.lchild.avg_mi/(self.lchild.visits+1) > (self.rchild.avg_mi/((self.rchild.visits+1)))):
                    self.rchild_switch = True
            #print(self.lchild.avg_mi/(self.lchild.visits+1),self.lchild.avg_mi/((self.lchild.visits+1)*5),self.rchild.avg_mi/(self.rchild.visits+1),self.rchild.avg_mi/((self.rchild.visits+1)*5),self.lchild_switch,self.rchild_switch)
            if(self.lchild.eval() > self.rchild.eval()):
                return self.lchild
            elif(self.rchild.eval() > self.lchild.eval()):
                return self.rchild
            else:
                #   if both are equal, then throw a random number
                if(np.random.randint(0,2,1)[0] == 0):
                    return self.lchild
                else:
                    return self.rchild


#   mutual information search tree
class MIST:

    def __init__(self,x,y,num_samples=1000,error=.05,n=10,disc_vars=[],weights=[],fraction=1.0):
        """
        Constructor for MIST object.
        """
        self.x = x
        self.y = y
        self.tree = None
        self.start_node = None
        self.best_score = 0
        self.paths = []
        self.best_path = []
        self.best_path_val = 0
        self.error = error
        self.num_samples = num_samples
        self.total_samples = len(x)
        #   list of elements for randomizing x
        self.sampler = [i for i in range(self.total_samples)]
        #   maximum mi from the set of variables (may not be knowable)
        self.mi_max_set = [i for i in range(len(x[0]))]
        self.mi_max = 0
        self.disc_vars = disc_vars
        self.weights = weights
        self.fraction = fraction
        #   mi from all variables
        self.mi_all = 0
        # if disc_vars != []:
        #     if weights != []:
        #         self.mi_all = MLP.mi_binary_discrete_weights(self.x,self.y,self.disc_vars,self.weights,self.fraction,k=1)
        #     else:
        #         self.mi_all = MLP.mi_binary_discrete(self.x,self.y,self.disc_vars,k=1)
        # else:
        #     if weights != []:
        #         self.mi_all = MLP.mi_binary_weights(self.x,self.y,self.weights,self.fraction,k=1)
        #     else:
        #         self.mi_all = MLP.mi_binary(self.x,self.y,k=1)
        self.mi_thr = self.mi_all
        if fraction >= 1.0:
            self.weight_s = 1.0/fraction
            self.weight_b = 1.0
        else:
            self.weight_s = 1.0
            self.weight_b = fraction
        self.n = n
        self.top_n = []


    #   compute mi using k=1 KSG with num_samples out of total_samples
    def get_mi(self,state):
        """
        Get MI using simple binary KSG on all events.
        """
        if(state == []):
            return 0
        else:
            #   randomize the input list to be used
            random.shuffle(self.sampler)
            temp_x = [[self.x[self.sampler[i]][j-1] for j in state] for i in range(self.num_samples)]
            temp_y = [self.y[self.sampler[i]] for i in range(self.num_samples)]
            return MLP.mi_binary(temp_x,temp_y,k=1)

    #   compute mi using k=1 KSG with the total sample
    def get_mi_total(self,state):
        if(state == []):
            return 0
        else:
            temp_x = [[self.x[i][j-1] for j in state] for i in range(self.total_samples)]
            temp_y = [self.y[i] for i in range(self.total_samples)]
            return MLP.mi_binary(temp_x,temp_y,k=1)

    #   compute mi using k=1 KSG with the total sample
    def get_mi_discrete_total(self,state):
        """
        Get MI using all events.
        """
        if(state == []):
            return 0
        else:
            temp_x = [[self.x[i][j-1] for j in state] for i in range(self.total_samples)]
            disc_list = [i for i in range(len(state)) if state[i] in self.disc_vars]
            temp_y = [self.y[i] for i in range(self.total_samples)]
            return MLP.mi_binary_discrete(temp_x,temp_y,disc_list,k=1)

    #   compute mi using k=1 KSG with num_samples out of total_samples
    def get_mi_discrete_weights(self,state):
        if(state == []):
            return 0
        else:
            #   randomize the input list to be used
            random.shuffle(self.sampler)
            temp_x = [[self.x[self.sampler[i]][j-1] for j in state] for i in range(self.num_samples)]
            temp_y = [self.y[self.sampler[i]] for i in range(self.num_samples)]
            disc_list = [i for i in range(len(state)) if state[i] in self.disc_vars]
            temp_weights = [self.weights[self.sampler[i]] for i in range(self.num_samples)]
            if disc_list != []:
                return MLP.mi_binary_discrete_weights(temp_x,temp_y,disc_list,temp_weights,self.fraction,k=1)
            else:
                return MLP.mi_binary_weights(temp_x,temp_y,temp_weights,self.fraction,k=1)

    #   compute mi using k=1 KSG with num_samples out of total_samples
    def get_mi_discrete_weights2(self,state):
        if(state == []):
            return 0
        else:
            #   randomize the input list to be used
            random.shuffle(self.sampler)
            temp_sampler = []
            i=0
            while len(temp_sampler) < self.num_samples:
                if self.y[self.sampler[i]] == [1.0]:
                    if self.weight_s*self.weights[self.sampler[i]] >= np.random.uniform(0,1,1)[0]:
                        temp_sampler.append(self.sampler[i])
                else:
                    if self.weight_b*self.weights[self.sampler[i]] >= np.random.uniform(0,1,1)[0]:
                        temp_sampler.append(self.sampler[i])
                i += 1
            temp_x = [[self.x[temp_sampler[i]][j-1] for j in state] for i in range(self.num_samples)]
            temp_y = [self.y[temp_sampler[i]] for i in range(self.num_samples)]
            disc_list = [i for i in range(len(state)) if state[i] in self.disc_vars]
            temp_weights = [self.weights[temp_sampler[i]] for i in range(self.num_samples)]
            if disc_list != []:
                return MLP.mi_binary_discrete(temp_x,temp_y,disc_list,k=1)
            else:
                return MLP.mi_binary(temp_x,temp_y,k=1)


    def get_mi_max(self):
        temp_x = [[self.x[i][j-1] for j in self.mi_max_set] for i in range(self.total_samples)]
        temp_y = [self.y[i] for i in range(self.total_samples)]
        self.mi_max = MLP.mi_binary_discrete_weights(temp_x,temp_y,disc_list=self.disc_vars,weights=self.weights,fraction=self.fraction,k=1)

    def get_mi_all(self):
        temp_x = [[self.x[i][j] for j in range(len(self.x[0]))] for i in range(self.total_samples)]
        temp_y = [self.y[i] for i in range(self.total_samples)]
        self.mi_all = MLP.mi_binary(temp_x,temp_y,k=1)
        return self.mi_all

    def traverse_tree_total(self,start_node):
        global total_visits
        if(start_node.states_left == []):
            start_node.end_state = True
            start_node.end_state_mi = self.get_mi_total(start_node.state)
            start_node.num_visits += 1
            total_visits += 1
            if(start_node.val > 0):
                start_node.parent.lchild_visits += 1
            else:
                start_node.parent.rchild_visits += 1
            return start_node.state,start_node.val
        else:
            return self.traverse_tree_total(start_node.choose_child())

    #   traverse tree discrete
    def traverse_tree_random(self,start_node):
        global total_visits
        if(start_node.states_left == []):
            start_node.end_state = True
            start_node.end_state_mi = self.get_mi(start_node.state)
            start_node.num_visits += 1
            total_visits += 1
            if(start_node.val > 0):
                start_node.parent.lchild_visits += 1
            else:
                start_node.parent.rchild_visits += 1
            return start_node.state,start_node
        else:
            return self.traverse_tree_random(start_node.choose_child())

    #   traverse tree with weights
    def traverse_tree_discrete_weights(self,start_node):
        global total_visits
        if(start_node.states_left == []):
            start_node.end_state = True
            start_node.end_state_mi = self.get_mi_discrete_weights(start_node.state)
            start_node.num_visits += 1
            total_visits += 1
            if(start_node.val > 0):
                start_node.parent.lchild_visits += 1
            else:
                start_node.parent.rchild_visits += 1
            return start_node.state,start_node
        else:
            return self.traverse_tree_discrete_weights(start_node.choose_child())

#   traverse tree with weights
    def traverse_tree_discrete_weights2(self,start_node):
        global total_visits
        if(start_node.states_left == []):
            start_node.end_state = True
            start_node.end_state_mi = self.get_mi_discrete_weights2(start_node.state)
            start_node.num_visits += 1
            total_visits += 1
            if(start_node.val > 0):
                start_node.parent.lchild_visits += 1
            else:
                start_node.parent.rchild_visits += 1
            return start_node.state,start_node
        else:
            return self.traverse_tree_discrete_weights2(start_node.choose_child())

    #   backprop for updating node statistics
    def backprop(self,node,val,win):
        if(node.parent.val==0):
            node.avg_mi += val
            node.visits += 1
            if(win == 1):
                node.num_wins += 1
        else:
            node.avg_mi += val
            node.visits += 1
            if(win == 1):
                node.num_wins += 1
            return self.backprop(node.parent,val,win)
    def add_path(self, path, path_mi):
        path = sorted(path)
        if len(self.top_n) < self.n:
            self.top_n.append([path, path_mi])
            self.top_n = sorted(self.top_n, key=lambda x:x[1], reverse=True)
        else:
            self.top_n.append([path, path_mi])
            self.top_n = sorted(self.top_n, key=lambda x:x[1], reverse=True)
            self.top_n = [self.top_n[i] for i in range(self.n)]


    # trace the tree by the path with the highest probability
    def highest_prob_path(self):
        start_tree = self.tree
        s = [i+1 for i in range(len(self.x[0]))]
        prob_path = []
        while(s != []):
            if(start_tree.lchild == None or start_tree.rchild == None):
                return prob_path
            if(start_tree.lchild.num_wins > start_tree.rchild.num_wins):
                prob_path.append(start_tree.lchild.val)
                s = remove_element(s,start_tree.lchild.val)
                start_tree = start_tree.lchild
            else:
                s = remove_element(s,-start_tree.rchild.val)
                start_tree = start_tree.rchild
        return sorted(prob_path)

    def run_mist_total(self,num_iters):
        global total_visits
        global total_playouts
        total_playouts = 0
        #   construct the empty start node
        s = [i+1 for i in range(len(self.x[0]))]
        self.tree = Node(0,None,[],s)
        self.start_node = self.tree
        #   now iterate over each layer
        #   find the best node for each layer
        total_visits = 0
        for j in range(num_iters):
            total_playouts += 1
            #print(self.best_path_val)
            path,winner = self.traverse_tree_total(self.start_node)
            if winner.end_state == True:
                path_mi = winner.end_state_mi
            else:
                path_mi = self.get_mi_total(path)
            print("Path %s of %s; %s; MI = %s" % (j,num_iters,path,path_mi))
            self.add_path(path,path_mi)
            self.paths.append(path)
            if(path_mi > self.mi_thr):
                self.best_path = path
                self.best_path_val = path_mi
                self.mi_thr = path_mi
            if(path_mi > (self.mi_thr-self.error*self.mi_thr)):
                self.backprop(winner,path_mi,1)
            else:
                self.backprop(winner,path_mi,0)
        self.best_path.sort()

    def run_mist_random(self,num_iters):
        global total_visits
        global total_playouts
        total_playouts = 0
        #   construct the empty start node
        s = [i+1 for i in range(len(self.x[0]))]
        self.tree = Node(0,None,[],s)
        self.start_node = self.tree
        #   now iterate over each layer
        #   find the best node for each layer
        total_visits = 0
        for j in range(num_iters):
            total_playouts += 1
            #print(self.best_path_val)
            path,winner = self.traverse_tree_random(self.start_node)
            if winner.end_state == True:
                path_mi = winner.end_state_mi
            else:
                path_mi = self.get_mi(path)
            print("Path %s of %s; %s; MI = %s" % (j,num_iters,path,path_mi))
            self.add_path(path,path_mi)
            self.paths.append(path)
            if(path_mi > self.mi_thr):
                self.best_path = path
                self.best_path_val = path_mi
                self.mi_thr = path_mi
            if(path_mi > (self.mi_thr-self.error*self.mi_thr)):
                self.backprop(winner,path_mi,1)
            else:
                self.backprop(winner,path_mi,0)
        self.best_path.sort()

    def run_mist_discrete_weights(self,num_iters):
        global total_visits
        global total_playouts
        print(self.disc_vars)
        total_playouts = 0
        #   construct the empty start node
        s = [i+1 for i in range(len(self.x[0]))]
        self.tree = Node(0,None,[],s)
        self.start_node = self.tree
        #   now iterate over each layer
        #   find the best node for each layer
        total_visits = 0
        for j in range(num_iters):
            total_playouts += 1
            #print(self.best_path_val)
            path,winner = self.traverse_tree_discrete_weights(self.start_node)
            if winner.end_state == True:
                path_mi = winner.end_state_mi
            else:
                path_mi = self.get_mi_discrete_weights(path)
            print("Path %s of %s; %s; MI = %s" % (j,num_iters,path,path_mi))
            self.add_path(path,path_mi)
            self.paths.append(path)
            if(path_mi > self.mi_thr):
                self.best_path = path
                self.best_path_val = path_mi
                self.mi_thr = path_mi
            if(path_mi > (self.mi_thr-self.error*self.mi_thr)):
                self.backprop(winner,path_mi,1)
            else:
                self.backprop(winner,path_mi,0)
        self.best_path.sort()

    def run_mist_discrete_weights2(self,num_iters):
        global total_visits
        global total_playouts
        print(self.disc_vars)
        total_playouts = 0
        #   construct the empty start node
        s = [i+1 for i in range(len(self.x[0]))]
        self.tree = Node(0,None,[],s)
        self.start_node = self.tree
        #   now iterate over each layer
        #   find the best node for each layer
        total_visits = 0
        for j in range(num_iters):
            total_playouts += 1
            #print(self.best_path_val)
            path,winner = self.traverse_tree_discrete_weights2(self.start_node)
            if winner.end_state == True:
                path_mi = winner.end_state_mi
            else:
                path_mi = self.get_mi_discrete_weights2(path)
            print("Path %s of %s; %s; MI = %s" % (j,num_iters,path,path_mi))
            self.add_path(path,path_mi)
            self.paths.append(path)
            if(path_mi > self.mi_thr):
                self.best_path = path
                self.best_path_val = path_mi
                self.mi_thr = path_mi
            if(path_mi > (self.mi_thr-self.error*self.mi_thr)):
                self.backprop(winner,path_mi,1)
            else:
                self.backprop(winner,path_mi,0)
        self.best_path.sort()

    def PCA_proj(self):
        vecs = []
        for i in range(len(self.top_n)):
            temp_vec = [0 for i in range(len(self.x[0])+1)]
            for j in range(len(self.top_n[i][0])):
                temp_vec[self.top_n[i][0][j]-1] = 1
            temp_vec[-1] = self.top_n[i][1]
            vecs.append(temp_vec)
        pca = MDS(n_components=2)
        pca.fit(vecs)
        transform = pca.fit_transform(vecs)
        x = [transform[i][0] for i in range(len(transform))]
        y = [transform[i][1] for i in range(len(transform))]
        c = [vecs[i][-1] for i in range(len(vecs))]
        max_c = max(c)
        c = [c[i]/max_c for i in range(len(c))]
        fig, axs = plt.subplots()
        axs.scatter(x,y,c=c)
        plt.show()



if __name__ == "__main__":

    N = 10000
    #   this is a test for the MIST algorithm

    #   lets try a signal-background problem consisting of gaussians
    #   redundant variables and noise
    sig_mus = [1.0,1.0,1.0,1.0,1.0]
    back_mus = [-1.0,-1.0,-1.0,-1.0,-1.0]
    sig_vars = [1.0,1.0,1.0,1.0,1.0]
    back_vars = [1.0,1.0,1.0,1.0,1.0]

    # one Gaussian variable separated by delta_mi = 2 for the third variable.
    # all others are uniform noise
    x_1 = [[np.random.uniform(0,1,1)[0],np.random.uniform(0,1,1)[0],np.random.normal(1,1,1)[0],np.random.uniform(0,1,1)[0]] for i in range(N)]
    y_1 = [[1.0] for i in range(N)]
    y_2 = [[-1.0] for i in range(N)]
    x_2 = [[np.random.uniform(0,1,1)[0],np.random.uniform(0,1,1)[0],np.random.normal(-1,1,1)[0],np.random.uniform(0,1,1)[0]] for i in range(N)]

    # generate the data and answer variables
    x = np.concatenate((x_1,x_2),axis=0)
    y = np.concatenate((y_1,y_2),axis=0)

    # create a MIST instance
    m = MIST(x,y)
    # run the mist algorithm for 20 iterations
    # using a random set of events for each playout
    m.run_mist_random(100)
    print(m.best_path_val)
