
import pickle 
from models.unet import HyperUnet
from kerastuner.tuners import Hyperband


def tuner_save(tuner,ckpt_path):

    # get best params
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        
    # save into pickle for later usage 
    try: 
        with open(ckpt_path+'/tuner.pkl', 'wb') as file:
            pickle.dump(tuner, file)
        with open(ckpt_path+'/best_hyper_parameters.pkl', 'wb') as file:
            pickle.dump(best_hyperparameters, file)
    except:
        print("Could not save to pickle")
    else:
        print(best_hyperparameters)

def train_keras_tuner(args,ckpt_path,img_size,channel_size,train_gen,val_gen):

    # build model 
    hypermodel = HyperUnet(img_size, channel_size)  

    # init tuner 
    tuner = Hyperband(
        hypermodel,
        max_epochs=args.maxepochs,
        objective="val_loss",
        seed=1,
        executions_per_trial=args.executions_per_trial,
        directory=ckpt_path + '_hyperband',
        project_name='test_kerastuner',
    )       
    
    
    # print tuner summary 
    tuner.search_space_summary()    
    input('Press any key to continue')
    
    # run tuner 
    tuner.search(train_gen, epochs=20, validation_data=val_gen)
    # tuner.search(x, y, epochs=30, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])
    #tuner.search(x_train, y_train, epochs=args.epochs, validation_split=0.1)
    #model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks, initial_epoch=initial_epoch)

    # show a summary of the search
    tuner.results_summary()

    # get the top model after 40 epochs 
    best_model = tuner.get_best_models(num_models=1)[0]

    # save the best model explicitly 
    best_model.save(ckpt_path+'/best_kerastuner_model')
    
    # save tuner 
    tuner_save(tuner,ckpt_path)
    
    return best_model