import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch_optimizer as optim
from torch.optim import SGD

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

######################## Model Class ########################

def evaluate_models(x_train, x_test, y_train, y_test):
    
    ensemble_models = {}

    models_and_params = {
        'Decision Tree': (Pipeline([
            ('classifier', DecisionTreeClassifier())
        ]), {
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }),
        
        'Random Forest': (Pipeline([
            ('classifier', RandomForestClassifier())
        ]), {
            'classifier__n_estimators': [100, 50, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }),
        
        'K-Nearest Neighbors': (Pipeline([
            ('classifier', KNeighborsClassifier())
        ]), {
            'classifier__n_neighbors': [5, 3, 7, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [2, 1]
        }),
        
        'XGBoost': (Pipeline([
            ('classifier', XGBClassifier( eval_metric = 'logloss' ) )
        ]), {
            'classifier__n_estimators': [100, 50, 200],
            'classifier__learning_rate': [0.1, 0.01, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }),

        'Support Vector Classifier': (Pipeline([
            ('classifier', SVC())
        ]), {
            'classifier__C': [1, 0.1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        })
    }

    def custom_weighted_score(y_true, y_pred):

        tn, fp, fn, tp = confusion_matrix( y_true, y_pred ).ravel()
        penalty_score = ( fn * 50 ) + ( fp * 5 )
        return -penalty_score

    # Create a scorer object
    penalty_scorer = make_scorer( custom_weighted_score )
    
    for model_name, ( model_pipeline, param_grid ) in models_and_params.items():

        print( f"Evaluating {model_name} with Hyperparameter Search" )
        
        grid_search = GridSearchCV( model_pipeline, param_grid, cv = 5, scoring = penalty_scorer, n_jobs = -1 )
        grid_search.fit( x_train, y_train )
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict( x_test )

        ensemble_models[model_name] = {
            'model': best_model,
            'report': classification_report( y_test, y_pred )
        }
        
        print(f'Best score for {model_name}: {grid_search.best_score_}')
        print(f'Best Parameters for {model_name}: {grid_search.best_params_}')
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'Precision: {precision_score(y_test, y_pred):.4f}')
        print(f'Recall: {recall_score(y_test, y_pred):.4f}')
        print(f'F1 Score: {f1_score(y_test, y_pred):.4f}')
        print(f'Report:\n{classification_report(y_test, y_pred)}')
        print("-" * 50)
    
    return ensemble_models

class Module(nn.Module):

    def __init__(self):

        super(Module, self).__init__()

        self.t_step = 0

    def save_training(self, folder):
        
        if not os.path.isdir( folder ):
            os.mkdir( folder )
        
        checkpoint = {
            'epoch': self.t_step,
            'model_state': self.state_dict()
        }

        files = os.listdir( folder )
        chpts = [ int( f.split('_')[1].replace('.pth', '') ) for f in files if f.startswith('checkpoint') ]
        if len( chpts ) > 50:
            if chpts[0] > self.t_step:
                for f in chpts:
                    file_path = os.path.join( folder, f'checkpoint_{f}.pth' )
                    os.remove( file_path )
                    print("\n\n=======================================================================================\n")
                    print("Deleting : {}, by maximum number of checkpoints".format( f ) )
                    print("\n=======================================================================================\n\n")
            else:
                chpts.sort()
                for f in chpts[:len(chpts) - 5]:
                    file_path = os.path.join( folder, f'checkpoint_{f}.pth' )
                    os.remove( file_path )
                    print("\n\n=======================================================================================\n")
                    print("Deleting : {}, by maximum number of checkpoints".format( f ) )
                    print("\n=======================================================================================\n\n")
        
        torch.save( checkpoint, folder + f'/checkpoint_{self.t_step}.pth' )
        print("\n\n=======================================================================================\n")
        print("Saved to: {} - {}".format(folder, self.t_step))
        print("\n=======================================================================================\n\n")

    def load_training(self, folder, chpt=None, eval=True):

        def get_max_checkpoint():
            files = os.listdir( folder )
            chpt = np.max( [ int( f.split('_')[1].replace('.pth', '') ) for f in files if f.startswith('checkpoint') ] )
            return chpt
        
        if not os.path.exists( folder ):

            print("\n\n=======================================================================================\n")
            print("Model not found, initializing randomly!")
            print("\n=======================================================================================\n\n")

        else:

            if not chpt is None:
                if os.path.isfile( folder + f'/checkpoint_{chpt}.pth' ): chpt = chpt
                else: chpt = get_max_checkpoint()
            else: chpt = get_max_checkpoint()
            
            print( folder + f'/checkpoint_{chpt}.pth' )

            cpt = torch.load( folder + f'/checkpoint_{chpt}.pth' )

            model_dict = self.state_dict()
            pretrained_dict = cpt['model_state']

            # Filter out matched parameters
            pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # Identify unmatched parameters
            unmatched_pretrained = {k for k in pretrained_dict if k not in model_dict}
            unmatched_current = {k for k in model_dict if k not in pretrained_dict}

            # Print unmatched parameters
            print("\nUnmatched parameters in the pretrained model:")
            for param in unmatched_pretrained:
                print(param)

            print("\nUnmatched parameters in the current model:")
            for param in unmatched_current:
                print(param)

            print("\n\n=======================================================================================\n")
            model_dict.update( pretrained_dict_filtered )
            print( self.load_state_dict( model_dict ) )
            
            self.t_step = cpt['epoch']
            
            print("\n\n=======================================================================================\n")
            print("Loaded from: {} - {}".format(folder, self.t_step))
            print("\n=======================================================================================\n\n")

class SAM(optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
    
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
    
        defaults = dict( rho = rho, **kwargs )
        super(SAM, self).__init__( params, defaults )
        self.base_optimizer = base_optimizer( self.param_groups, **kwargs )
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / ( grad_norm + 1e-12 )
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['e_w'] = p.grad * scale
                p.add_(self.state[p]['e_w'])  # ascent step

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # descent step

        self.base_optimizer.step()  # apply optimizer step

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        
        shared_device = self.param_groups[0]["params"][0].device
        
        norm = torch.norm(
            torch.stack( [
                p.grad.norm( p = 2 ).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ] ),
            p = 2
        )
        return norm

class LogisticRegressionModel(Module):
    
    def __init__(self, input_size, log_dir, name, num_groups=1, device='cpu'):
        
        super(LogisticRegressionModel, self).__init__()
        self.log_dir_base = log_dir
        self.log_dir = log_dir + name
        self.input_size = input_size
        self.num_groups = num_groups
        self.device = device

        self.w = nn.Parameter( torch.randn( num_groups, input_size, 1, device = device ) ).to( device ) # weights for the logistic regression
        self.b = nn.Parameter( torch.ones( num_groups, 1, device = device ) ).to( device ) # bias for the logistic regression

        self.final_w = nn.Parameter( torch.randn( 6, 2, device = device ) ).to( device ) # weights for the ensemble
        self.final_b = nn.Parameter( torch.ones( 2, device = device ) ).to( device ) # bias for the ensemble
    
    def forward(self, x):
        """
            features = [bs, feature_size], float
            group_number = [bs], int
        """

        features, group_number, ensemble = x  # Unpack input
        feature_size = features.size(1)
        
        # Reshape group_number to match the first dimension of w and b
        group_number_expanded = group_number.view( -1, 1, 1 ).expand( -1, feature_size, 1 )
        
        # Gather group-specific weights and biases for each sample
        w_selected = torch.gather( self.w, 0, group_number_expanded).squeeze(-1) # [bs, feature_size]
        b_selected = torch.gather( self.b, 0, group_number.view( -1, 1 ) ).squeeze(-1) # [bs]
        
        # Compute the logistic regression on the Inputs
        if self.training: features = F.dropout( features, p = 0.2 )
        output_logistic_regression = torch.sigmoid( torch.sum( features * w_selected, dim = 1 ) + b_selected )

        # Final Output
        final_input = torch.cat( [ output_logistic_regression.view( -1, 1 ), ensemble ], dim = 1 )
        if self.training: final_input = F.dropout( final_input, p = 0.2 )
        output = torch.matmul( final_input, self.final_w ) + self.final_b

        if self.training: return output_logistic_regression, output

        return torch.softmax( output, dim = 1 )[:,1]

    """
        target = 1
        pred = 0

        distance = target - pred
        weight = ( ( distance > 0 ) * 50 ) + ( ( distance < 0 ) * 5 )
        scale = abs( distance ) * weight
        print( weight, scale )

        target = 0
        pred = 1

        distance = target - pred
        weight = ( ( distance > 0 ) * 50 ) + ( ( distance < 0 ) * 5 )
        scale = abs( distance ) * weight
        print( weight, scale )
    """
    def weighted_binary_cross_entropy_single(self, y_hat, y, epsilon=1e-10):
        
        fp_weight = 5.0
        fn_weight = 50.0

        with torch.no_grad():
            distance = torch.abs( y - y_hat ) # Compute the distance between the target and the prediction to unserstand if it is a false positive or a false negative
            weight = torch.where( ( y == 1 ) & ( y_hat < 0.5 ), fp_weight, torch.where( ( y == 0 ) & ( y_hat > 0.5 ), fn_weight, 1.0 ) ) # Compute the weight for the loss
            sclaed_loss = distance * weight # Cpompute the loss sacle factor

        base_loss = -y * torch.log( y_hat + epsilon ) - ( 1 - y ) * torch.log( ( 1 - y_hat ) + epsilon ) # base bce loss
        weighted_loss = ( 1 + sclaed_loss ) * base_loss # Scale the base loss
        
        return torch.mean( weighted_loss )

    def weighted_binary_cross_entropy_double(self, y_hat, y, epsilon=1e-10):
       
        fp_weight = 5.0
        fn_weight = 50.0
        
        log_probs = F.log_softmax( y_hat, dim = 1 )
        
        y = y.long()
        log_prob_true = log_probs[range(y.size(0)), y]  # select the log prob of the correct class
        
        with torch.no_grad():

            probs = torch.exp( log_probs[:, 1] )
            distance = torch.abs( y - probs )

            weight = torch.where( ( y == 1 ) & ( probs < 0.5 ), fp_weight,
                                  torch.where( ( y == 0 ) & ( probs > 0.5 ), fn_weight, 1.0 ) )
            
            scaled_loss = distance * weight

        weighted_loss = -( 1 + scaled_loss ) * log_prob_true

        # Return the mean of the weighted loss
        return torch.mean( weighted_loss )

    def __create_balanced_loader__(self, x, t, batch_size):
        
        # Compute class weights
        class_sample_counts = torch.bincount(t)
        weights = 1. / class_sample_counts
        sample_weights = weights[t]

        # Create sampler
        sampler = WeightedRandomSampler( sample_weights, len( sample_weights ), replacement = True )

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset( x[0], x[1].to( torch.long ), x[2], t )
        loader = DataLoader( dataset, batch_size = batch_size, sampler = sampler )
        
        return loader

    def fit(self, x, t, epochs=1000, lr=1e-3, batch_size=32):

        self.train()

        optimizer = torch.optim.SGD( self.parameters(), lr = lr, momentum = 0.9, weight_decay = 1e-6 ) # L2 regularization

        loss_history = []

        epoch_bar = tqdm( range( epochs ), desc =  "Epochs", unit = "epoch" )

        for epoch in epoch_bar:
            
            # conver t to long tensor
            mini_batch_loader = self.__create_balanced_loader__( x, t.to( torch.long ), batch_size )

            # Initialize a variable to track the total loss for the epoch
            epoch_loss = 0.0

            # Training loop with mini-batches
            for x_batch, t_batch in mini_batch_loader:

                y = self.forward( x_batch ) # logistic regression forward pass
                y = torch.squeeze( y, dim = -1 ) # remove the last dimension
                loss = self.weighted_binary_cross_entropy( y, t_batch ) # loss calculation
                l1_loss = torch.norm( self.w, 1 ) # L1 regularization
                loss += 1e-3 * l1_loss # add L1 regularization to the loss
                
                loss.backward() # backpropagation gradient calculation
                optimizer.step() # update the weights
                optimizer.zero_grad() # reset the gradients

                self.t_step += 1 # increment the training step counter

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len( mini_batch_loader )
            loss_history.append( avg_epoch_loss )

            epoch_bar.set_postfix( loss = avg_epoch_loss )

    def fit_sam(self, x, t, epochs=1000, lr=1e-3, rho=0.05, batch_size=32):
    
        self.train()

        # Initialize SAM optimizer with SGD as the base optimizer
        base_optimizer = SGD
        optimizer = SAM( self.parameters(), base_optimizer = base_optimizer, lr = lr, momentum = 0.9, weight_decay = 1e-6, rho = rho )

        loss_history = []
        epoch_bar = tqdm( range( epochs ), desc = "Epochs", unit = "epoch" )

        for epoch in epoch_bar:
            
            # Convert target tensor to long type if necessary
            mini_batch_loader = self.__create_balanced_loader__( x, t.to( torch.long ), batch_size )

            # Initialize a variable to track the total loss for the epoch
            epoch_loss = 0.0

            # Training loop with mini-batches
            for x_batch, g_batch, e_batch, t_batch in mini_batch_loader:
                
                # Forward pass
                y = self( ( x_batch, g_batch, e_batch ) )
                loss = self.weighted_binary_cross_entropy_single( y[0], t_batch ) * 1e-3
                loss += self.weighted_binary_cross_entropy_double( y[1], t_batch )
                
                # Step 1: First SAM "ascent" step
                loss.backward()  # Compute gradients and retain graph
                optimizer.first_step( zero_grad = True ) # Take a step in gradient direction

                # Step 2: Calculate the loss again after moving parameters in ascent direction
                y = self( ( x_batch, g_batch, e_batch ) )
                loss = self.weighted_binary_cross_entropy_single( y[0], t_batch ) * 1e-3
                loss += self.weighted_binary_cross_entropy_double( y[1], t_batch )
                
                # Second SAM "descent" step
                loss.backward()  # Compute gradients again without retaining graph
                optimizer.second_step(zero_grad=True)  # Take the SAM step to minimize sharpness

                # Accumulate loss for this batch
                epoch_loss += loss.item()

            # Calculate and record the average loss for the epoch
            avg_epoch_loss = epoch_loss / len( mini_batch_loader )
            loss_history.append( avg_epoch_loss )

            # Update progress bar
            epoch_bar.set_postfix(loss=avg_epoch_loss)

        return loss_history

    def predict(self, x):
        self.eval()
        return self.forward( x ).cpu().detach().numpy()
    
    def k_fold_cross_validation_fit(self, x_train, y_train, k=5, epochs=1000, lr=1e-3, rho=0.05,  batch_size=32):

        x_train_tensor = torch.tensor( x_train[0], dtype = torch.float32 )
        x_group_tensor = torch.tensor( x_train[1], dtype = torch.float32 )
        x_ensemble_tensor = torch.tensor( x_train[2], dtype = torch.float32 )
        y_train_tensor = torch.tensor( y_train, dtype = torch.float32 )

        skf = StratifiedKFold( n_splits = k, shuffle = True, random_state = 42 )  # 5-fold cross-validation stratisfied by the target variable

        fold_results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "state_dict": []
        }

        # Execute the cross-validation loop
        fold = 1
        for train_index, val_index in skf.split( x_train_tensor, y_train_tensor ):

            print( f"Fold {fold}" )

            # Split the data for this fold
            x_train_fold, x_val_fold = x_train_tensor[train_index].to( self.device ), x_train_tensor[val_index].to( self.device )
            x_group_fold, x_group_val_fold = x_group_tensor[train_index].to( self.device ), x_group_tensor[val_index].to( self.device )
            x_ensemble_fold, x_ensemble_val_fold = x_ensemble_tensor[train_index].to( self.device ), x_ensemble_tensor[val_index].to( self.device )
            y_train_fold, y_val_fold = y_train_tensor[train_index].to( self.device ), y_train_tensor[val_index].to( self.device )

            # Initialize the model
            model = LogisticRegressionModel( self.input_size, '', '', self.num_groups, device = self.device )

            # Train the model
            model.fit_sam( ( x_train_fold, x_group_fold, x_ensemble_fold ),
                             y_train_fold, epochs = epochs, lr = lr, rho = rho, batch_size = batch_size )

            # Evaluate on the validation set
            y_hat_val = np.round( model.predict( ( x_val_fold, x_group_val_fold.to( torch.long ), x_ensemble_val_fold ) ) )

            # Convert predictions and true values to numpy arrays for metrics calculation
            y_val_np = y_val_fold.detach().cpu().numpy()
            y_hat_val_np = y_hat_val

            # Calculate and store metrics
            accuracy = accuracy_score( y_val_np, y_hat_val_np )
            precision = precision_score( y_val_np, y_hat_val_np )
            recall = recall_score( y_val_np, y_hat_val_np )
            f1 = f1_score( y_val_np, y_hat_val_np )

            fold_results["accuracy"].append( accuracy )
            fold_results["precision"].append( precision )
            fold_results["recall"].append( recall )
            fold_results["f1"].append( f1 )
            fold_results["state_dict"].append( model.state_dict() )

            print( f"Fold {fold} Results:" )           

            print( f"Num High Risk predicted as Low Risk: {len( y_hat_val_np[ ( y_hat_val_np == 0 ) & ( y_val_np == 1 ) ] ) }" )
            print( f"Num High Risk predicted as High Risk: {len( y_hat_val_np[ ( y_hat_val_np == 1 ) & ( y_val_np == 1 ) ] ) }\n" )

            print( f"Num Low Risk predicted as Low Risk: {len( y_hat_val_np[ ( y_hat_val_np == 0 ) & ( y_val_np == 0 ) ] ) }" )
            print( f"Num Low Risk predicted as High Risk: {len( y_hat_val_np[ ( y_hat_val_np == 1 ) & ( y_val_np == 0 ) ] ) }\n" )
            
            print( f"Accuracy: {accuracy}" )
            print( f"Precision: {precision}" )
            print( f"Recall: {recall}" )
            print( f"F1 Score: {f1}" )

            fold += 1

        # After all folds, calculate average results
        avg_accuracy = np.mean( fold_results["accuracy"] )
        avg_precision = np.mean( fold_results["precision"] )
        avg_recall = np.mean( fold_results["recall"] )
        avg_f1 = np.mean( fold_results["f1"] )

        # Display average results
        print("\nCross-Validation Results (Averaged):")
        print(f"Average Accuracy: {avg_accuracy}")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}")
        print(f"Average F1 Score: {avg_f1}")

        # Use all folds to update the model
        self.t_step = model.t_step * k
        update_state_dict = {}
        for key in self.state_dict().keys():
            update_state_dict[key] = torch.mean( torch.stack( [ torch.tensor( fold_results["state_dict"][i][key] ) for i in range( k ) ], dim = 0 ), dim = 0 )        
        self.load_state_dict( update_state_dict )

    def save(self):
        self.save_training( self.log_dir )
    
    def load(self, chpt=None, eval=True):
        self.load_training( self.log_dir, chpt, eval )

######################## Data Processing ########################

def one_hot_encode_column(column, num_classes):
    one_hot = np.zeros( ( len( column ), num_classes ) )
    for i, val in enumerate( column ):
        if not np.isnan( val ): one_hot[ i, int( val ) ] = 1
    return one_hot

class DataProcessing(object):

    def __init__(self, config_name, dataset, one_hot_encoding=False):
        
        self.one_hot_encoding = one_hot_encoding

        # columns configurations
        self.categorical_config = pd.read_csv(f'./{config_name}/Book(Categorical).csv')
        self.numerical_config = pd.read_csv(f'./{config_name}/Book(Numerical).csv')
        self.target_columns = [ "CLASS", "CLUSTER" ]
        self.id_column = "ORDER_ID"

        # dataset loader
        self.dataset = dataset.copy()
        year = datetime.datetime.now().year
        self.dataset["AGE"] = year - pd.to_datetime(self.dataset["B_BIRTHDATE"]).dt.year
        self.dataset["AGE"].fillna( self.dataset["AGE"].mean(), inplace = True )
        self.dataset["VALID_YEAR"] = self.dataset['Z_CARD_VALID'].apply( lambda x: str(x).split('.')[1] if x is not None else None )
        self.dataset["VALID_MONTH"] = self.dataset['Z_CARD_VALID'].apply( lambda x: str(x).split('.')[0] if x is not None else None )
        self.dataset["HOUR_ORDER"] = self.dataset['TIME_ORDER'].apply( lambda x: str(x).split(':')[0] if x is not None else None )
        self.dataset['DATE_LORDER'] = pd.to_datetime(self.dataset['DATE_LORDER'], errors='coerce')
        self.dataset['VALID_DATE'] = pd.to_datetime( self.dataset['VALID_YEAR'] + '-' + self.dataset['VALID_MONTH'] + '-01', errors = 'coerce' )
        self.dataset['DAYS_TO_EXPIRE_CARD'] = (self.dataset['VALID_DATE'] - self.dataset['DATE_LORDER']).dt.days
        self.dataset['DAYS_TO_EXPIRE_CARD'].fillna( -1, inplace = True )
        self.dataset['WEEKDAY_ORDER'].fillna( 'Unknown', inplace = True )
        self.dataset['NO_ITEMS'] = self.dataset[ [ 'ANUMMER_01', 'ANUMMER_02', 'ANUMMER_03', 'ANUMMER_04', 
                                                   'ANUMMER_05', 'ANUMMER_06', 'ANUMMER_07', 'ANUMMER_08', 
                                                   'ANUMMER_09', 'ANUMMER_10' ] ].notnull().sum( axis = 1 )

        df_long = self.dataset.melt( id_vars = ['ORDER_ID'], 
                                     value_vars = [ 'ANUMMER_01', 'ANUMMER_02', 'ANUMMER_03', 'ANUMMER_04', 'ANUMMER_05', 'ANUMMER_06', 'ANUMMER_07', 'ANUMMER_08', 'ANUMMER_09', 'ANUMMER_10' ], 
                                     var_name = 'ANUMMER_INDEX', 
                                     value_name = 'ANUMMER' )

        df_long = df_long.drop( columns = [ 'ANUMMER_INDEX' ] )
        df_long = df_long.drop_duplicates( subset = [ 'ORDER_ID', 'ANUMMER' ] )
        df_long = df_long.reset_index( drop = True )
        df_long = df_long.merge( self.dataset, on = 'ORDER_ID', how = 'left' )

        self.categorical_config.loc[ self.categorical_config['Feature'].isin( [ 'ANUMMER_01', 'ANUMMER_02', 'ANUMMER_03', 'ANUMMER_04', 'ANUMMER_05', 'ANUMMER_06', 'ANUMMER_07', 'ANUMMER_08', 'ANUMMER_09', 'ANUMMER_10' ] ), 'Ignore' ] = True

        # fill any categorical missing values with 'Unknown'
        for col in self.categorical_config['Feature'].values:
            df_long[col].fillna( 'Unknown', inplace = True )

        self.dataset = df_long

        # label encoder
        with open( f'./{config_name}/label_encoders.pkl', 'rb' ) as f:
            self.label_encoder = pickle.load(f)

        # scaler
        with open( f'./{config_name}/standard_scaler.pkl', 'rb' ) as f:
            self.scaler = pickle.load(f)

        self.__process_outilers__()
        self.__encoder_categorical__()
        self.__process_missing_values__()
        self.__apply_log_trasnformation__()
        self.__apply_normalization__()

    def __encoder_categorical__(self):

        for row in self.categorical_config.iterrows():
            col = row[1]['Feature']
            self.dataset[col+'_encoded'] = self.label_encoder[col].transform( self.dataset[col].astype( str ) )

    def __process_outilers__(self):
        
        for row in self.numerical_config.iterrows():

            col = row[1]['Feature']
            upper_bound = row[1]['Outlier Upper Bound']
            lower_bound = row[1]['Outlier Lower Bound']
            method = row[1]['Outlier Treatment']

            if method == 'Remove':
                self.dataset = self.dataset[ ( self.dataset[col] >= lower_bound ) & ( self.dataset[col] <= upper_bound ) ]
            elif method == 'Replace with Mean':
                mean = self.dataset[col].mean()
                self.dataset[col] = np.where( self.dataset[col] < lower_bound, mean, self.dataset[col] )
                self.dataset[col] = np.where( self.dataset[col] > upper_bound, mean, self.dataset[col] )
            elif method == 'Replace with Median':
                median = self.dataset[col].median()
                self.dataset[col] = np.where( self.dataset[col] < lower_bound, median, self.dataset[col] )
                self.dataset[col] = np.where( self.dataset[col] > upper_bound, median, self.dataset[col] )
            elif method == 'Replace with Boundary':
                self.dataset[col] = np.where( self.dataset[col] < lower_bound, lower_bound, self.dataset[col] )
                self.dataset[col] = np.where( self.dataset[col] > upper_bound, upper_bound, self.dataset[col] )

    def __process_missing_values__(self):

        for row in self.categorical_config.iterrows():

            col = row[1]['Feature']
            method = row[1]['FillingMissingMethod']
            accuracy = row[1]['ModelAccuracy']
            ignore = row[1]['Ignore']

            if col in self.target_columns: continue # skip the target column
            if ignore: continue # skip the column if it is ignored
            if accuracy < 0.7: continue # skip the column if the model accuracy is less than 0.7

            missing_rows = self.dataset[ self.dataset[col] == 'Unknown' ]
            if len(missing_rows) == 0: continue

            if method == 'Mode':
                self.dataset = self.dataset[col+'_encoded'].fillna( self.dataset[col+'_encoded'].mode()[0] )

            elif method == 'NullClass':
                self.dataset[col+'_encoded'] = self.dataset[col+'_encoded'].fillna( self.label_encoder[col].transform( None ) )

            elif method == 'CategoryNB':

                has_model = row[1]['FillModel']
                if not has_model: 
                    self.dataset[col] = self.dataset[col].fillna('NullClass')
                    continue

                features = row[1]['SelectedFeatures'].split(',')
                features = [ (f+'_encoded').strip() for f in features ]
                model_path = row[1]['ModelPath']

                with open( model_path, 'rb' ) as f:
                    model = pickle.load(f)

                # fill the missing values with the predicted values
                missing = missing_rows[features]
                missing[col+'_encoded'] = model.predict( missing[features] )
                self.dataset.loc[ missing.index, col+'_encoded' ] = missing[col+'_encoded']
            
            elif method == 'KNN':

                has_model = row[1]['FillModel']
                if not has_model:
                    self.dataset[col] = self.dataset[col].fillna('NullClass')
                    continue

                features = row[1]['SelectedFeatures'].split(',')
                features = [f.strip() for f in features]
                features = [ f + '_encoded' if f in self.categorical_config['Feature'].values else f for f in features ]
                model_path = row[1]['ModelPath']
                scaler_path = row[1]['ModelScalerPath']

                # Load the KNN model and StandardScaler
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

                # Identify missing rows for prediction
                missing_rows = self.dataset[ ( self.dataset[col] == 'Unknown' ) | ( self.dataset[col].isnull() ) ]
                if missing_rows.empty:
                    continue

                # Scale the features of missing rows before prediction
                x_missing = missing_rows[features].fillna(0)  # Fill missing with 0 before scaling
                x_missing_scaled = scaler.transform(x_missing)
                
                # Fill the missing values with predicted values
                predicted_values = model.predict(x_missing_scaled)
                self.dataset.loc[missing_rows.index, col+'_encoded'] = predicted_values
    
    def __apply_log_trasnformation__(self):

        for row in self.numerical_config.iterrows():
            col = row[1]['Feature']
            ignore = row[1]['Ignore'] 
            if ignore: continue # skip the column if it is ignored
            if row[1]['ApplyLog'] == 1:
                self.dataset[col] = np.log( self.dataset[col] + 1 )

    def __apply_normalization__(self):

        num_column_names = self.numerical_config[ self.numerical_config['Ignore'] == 0 ]['Feature'].values # only numerical columns that are not ignored
        self.dataset[num_column_names] = self.scaler.transform( self.dataset[num_column_names] )

    def get_train_test(self, test_size=0.2, random_state=42, target='CLASS'):

        # Get categorical and numerical features
        all_usable_cat_features = self.categorical_config[ 
            ( ~self.categorical_config['Feature'].isin( self.target_columns ) ) & 
            ( self.categorical_config['Ignore'] == False ) ]
        categorical_features = all_usable_cat_features['Feature'].values
        categorical_features = [f + '_encoded' for f in categorical_features]
        categorical_features_number_classes = [ len( self.label_encoder[f].classes_ ) for f in all_usable_cat_features['Feature'].values ]
        numerical_features = self.numerical_config[ self.numerical_config['Ignore'] == 0 ]['Feature'].values

        train_df, test_df = train_test_split(
            self.dataset, test_size = test_size, random_state = random_state, stratify = self.dataset[target]
        )

        if len( categorical_features ) > 0:

            train_encoded_cat = [ one_hot_encode_column( train_df[feature].values, categorical_features_number_classes[i] ) 
                                for i, feature in enumerate( categorical_features ) ]
            
            test_encoded_cat = [ one_hot_encode_column( test_df[feature].values, categorical_features_number_classes[i] )
                                for i, feature in enumerate( categorical_features ) ]

            x_train_cat = np.concatenate( train_encoded_cat, axis = 1 )
            x_test_cat = np.concatenate( test_encoded_cat, axis = 1 )

            x_train_num = train_df[numerical_features].values
            x_test_num = test_df[numerical_features].values

            x_train_one_hot = np.concatenate( [ x_train_cat, x_train_num ], axis = 1 )
            x_test_one_hot = np.concatenate( [ x_test_cat, x_test_num ], axis = 1 )

            x_train = np.concatenate( [ train_df[ categorical_features ], 
                                        train_df[ numerical_features ] ], axis = 1 )

            x_test = np.concatenate( [ test_df[ categorical_features ],
                                       test_df[ numerical_features ] ], axis = 1 )

            if target+'_encoded' in train_df.columns:
                y_train = train_df[target+'_encoded'].values
                y_test = test_df[target+'_encoded'].values
            else:
                y_train = train_df[target].values
                y_test = test_df[target].values

            ids_train = train_df[self.id_column].values
            ids_test = test_df[self.id_column].values

            return ( ids_train, ids_test ), ( x_train_one_hot, x_train ), ( x_test_one_hot, x_test ), y_train, y_test

        else:

            x_train_cat = train_df[ categorical_features ]
            x_train_num = train_df[ numerical_features ]
            x_train = np.concatenate( [ x_train_cat, x_train_num ], axis=1 )
        
            x_test_cat = test_df[ categorical_features ]
            x_test_num = test_df[ numerical_features ]
            x_test = np.concatenate( [ x_test_cat, x_test_num ], axis=1 )

            if target+'_encoded' in train_df.columns:
                y_train = train_df[target+'_encoded'].values
                y_test = test_df[target+'_encoded'].values
            else:
                y_train = train_df[target].values
                y_test = test_df[target].values

            ids_train = train_df[self.id_column].values
            ids_test = test_df[self.id_column].values

            return ( ids_train, ids_test ), x_train, x_test, y_train, y_test

    def get_inference_data(self, one_hot_encoding=True):

        all_usable_cat_features = self.categorical_config[ 
            ( ~self.categorical_config['Feature'].isin( self.target_columns ) ) & 
            ( self.categorical_config['Ignore'] == False ) ]
        categorical_features = all_usable_cat_features['Feature'].values
        categorical_features = [f + '_encoded' for f in categorical_features]
        categorical_features_number_classes = all_usable_cat_features['TotalClasses'].values.astype(int)
        numerical_features = self.numerical_config[ self.numerical_config['Ignore'] == 0 ]['Feature'].values

        if one_hot_encoding:

            if len( categorical_features ) > 0:
            
                encoded_cat = []
                for i, feature in enumerate( categorical_features ):
                    num_classes = categorical_features_number_classes[i]
                    encoded_cat.append( one_hot_encode_column( self.dataset[feature].values, num_classes ) )

                x_cat = np.concatenate( encoded_cat, axis = 1 )
                x_num = self.dataset[numerical_features].values
                x_data = np.concatenate( [ x_cat, x_num ], axis = 1 )
            
            else:

                x_data = self.dataset[numerical_features].values

        else:

            x_cat = self.dataset[categorical_features].values
            x_num = self.dataset[numerical_features].values
            x_data = np.concatenate([x_cat, x_num], axis=1)

        ids = self.dataset[self.id_column].values

        return ids, x_data

class HibriModel():

    def __init__(self, config_name):

        # load the segmentation model
        with open( f'./{config_name}/segmentation_model.pkl', 'rb' ) as f:
            self.logistic_segmentation_model = pickle.load( f )

        # load the ensemble models
        with open( f'./{config_name}/ensemble_models.pkl', 'rb' ) as file:
            self.ensemble_models = pickle.load( file )

        # load the Logistic Regression model
        self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.model = LogisticRegressionModel( 696, f'./{config_name}/logs/', 'logistic_regression', 76, device = self.device )
        self.model.load()
        self.model.to( self.device )

    def predict(self, x, selected_model='Ensemble'):

        x_id, x_onehot, x_label = x

        # Predict the segmentation classes
        segmentation_classes = self.logistic_segmentation_model.predict( x_onehot )

        # Add the segmentation classes to the features
        x_label = np.concatenate( [ segmentation_classes.reshape( -1, 1 ), x_label ], axis = 1 )

        if selected_model != 'Ensemble':
            model = self.ensemble_models[selected_model]['model']
            y_hat = model.predict( x_label )
            df = pd.DataFrame( { 'predicted': y_hat, 'ORDER_ID': x_id } )
            df = df.groupby( 'ORDER_ID' ).agg( { 'predicted': 'mean' } )
            return df.reset_index()

        # Predict using the ensemble models
        ensemble_predictions = []
        for model_name, model_data in tqdm( self.ensemble_models.items() ):
            model = model_data['model']
            y_pred = model.predict( x_label )
            ensemble_predictions.append( y_pred )
        ensemble_predictions = np.array( ensemble_predictions ).T

        x_test_tensor = torch.tensor( x_onehot, dtype = torch.float32 ).to( self.device )
        x_groups_tensor = torch.tensor( segmentation_classes, dtype = torch.long ).to( self.device )
        x_ensemble_tensor = torch.tensor( ensemble_predictions, dtype = torch.float32 ).to( self.device )

        y_hat = self.model.predict( ( x_test_tensor, x_groups_tensor, x_ensemble_tensor ) )

        df = pd.DataFrame( { 'predicted': y_hat, 'ORDER_ID': x_id } )
        df = df.groupby( 'ORDER_ID' ).agg( { 'predicted': 'mean' } )

        return df.reset_index()

        