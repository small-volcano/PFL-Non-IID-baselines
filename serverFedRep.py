from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
#from utils.data_utils import read_client_data
from utils.model_utils import read_data, read_user_data #added
from threading import Thread
import time


class FedRep(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        # select slow clients
        self.set_slow_clients()

        data = read_data(self.dataset) #added


        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            #train, test = read_client_data(dataset, i)
            id, train , test = read_user_data(i, data, dataset=self.dataset) #added
            client = clientAVG(device, i, train_slow, send_slow, train, test, model, batch_size, learning_rate, local_steps)
            self.clients.append(client)

        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            #self.send_models()
            self.send_parameters_fedrep()
            #self.send_parameters_fedbn()


            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.timestamp = time.time() #
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()
            curr_timestamp = time.time() #
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_clients)
            print("glob_iter: ",i,"    train_time: ",train_time)
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.timestamp = time.time() #
            self.receive_models()
            self.aggregate_parameters()
            curr_timestamp = time.time() #
            agg_time = curr_timestamp - self.timestamp
            print("glob_iter: ",i,"    agg_time: ",agg_time)

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()
