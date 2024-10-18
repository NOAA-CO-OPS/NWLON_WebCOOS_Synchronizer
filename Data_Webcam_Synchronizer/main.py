import datetime
import glob
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip, concatenate_videoclips
import numpy as np
import os
import pandas as pd
import pickle
import pywebcoos # Package available on the WebCOOS GitHub #
import random
from sklearn.neural_network import MLPClassifier

from .CoopsApi import CoopsApi
from .CoopsApiTwo import CoopsApiTwo

def view_cameras(token):
    '''
    Function to view available cameras on the WebCOOS API. Wrapper for the pywebcoos wrapper.
    
    Parameters
    _ _ _ _ _ 
    token : str
        WebCOOS API token.
    
    Returns
    _ _ _ _ _ 
    None, but prints a list.
      
    '''
    api = pywebcoos.API(token)
    api.view_cameras()


def view_products(camera,token):
    '''
    Function to view available cameras on the WebCOOS API. Wrapper for the pywebcoos wrapper.
    
    Parameters
    _ _ _ _ _ 
    camera : str
        The name of the camera from which to view available products.
    token : str
        WebCOOS API token.
    
    Returns
    _ _ _ _ _ 
    None, but prints info.
      
    '''    
    api = pywebcoos.API(token)
    webcoos.view_products(camera)


def view_product_inventory(camera,product,token):
    '''
    Function to view the inventory of a product from a WebCOOS camera. Wrapper for the pywebcoos wrapper.
    
    Parameters
    _ _ _ _ _ 
    camera : str
        The name of the camera from which to view the inventory.
    product : str
        The name of the imagery product from which to view the inventory.
    token : str
        WebCOOS API token.
    
    Returns
    _ _ _ _ _ 
    None, but prints info.
      
    '''       
    api =  pywebcoos.API(token)
    api.view_data(camera,product)


def synch(station,camera,data_product,camera_product,value,time_start,time_end,interval,cutoff,sep_model,token,save_dir):
    '''
    Function to synchronize webcamera imagery frames with CO-OPS data.
    
    Parameters
    _ _ _ _ _ 
    station : int
        The NWLON station ID from which to use data.
    camera : str
        The name of the camera from which to use imagery.
    data_product : str
        The CO-OPS data product to use from station (e.g. water_level).
    camera_product : str
        The name of the imagery product to use from camera.
    value : str or float
        A qualifier to determine the type of data from which a movie is made. Options are:
            'all': Make a movie of all data points in data_product between time_start and time_end.
            float e.g. 0.5: Make a movie of n data points in data_product between time_start and time_end that are ~equal to this value, where n is given by cutoff.
    time_start : str
        The time to begin synchronizing data, in local time at the camera. In format 'yyyymmddHHMM'.
    time_end : str
        The time to stop synchronizing data, in local time at the camera. In format 'yyyymmddHHMM'.
    interval : int
        The time interval for data points and camera imagery. If an interval is entered that is less than or greater than the downloaded station data interval, the data is interpolated to the desired interval.
    cutoff : int or None
        Crop available data to this many data points / frames.
        For example, if value='highest' and value=10, the 10 hishest data values between time_start and time_end will be synchronized.
    sep_model : str or None
        The full path to a saved view separation model created with ViewSeparator.
    token : str
        WebCOOS API token.
    save_dir : str
        The directory to which to save frames and the final movie.
        
    Returns
    _ _ _ _ _ 
    datas : Pandas DataFrame
        The synchronized data and webcamera frames. Each data point in datas contains a time, data value, saved image path, and (if applicable) predicted view number. 
    '''           
    # First get all the data of the requested product for the requested time period #
    data = _call_CoopsApi(station,data_product,time_start,time_end)
    data = data[data['date_time']>=_datestr2dt(time_start)]
    data = data[data['date_time']<=_datestr2dt(time_end)]
    
    # Interpolate to the desired data interval #
    data_dates = pd.date_range(_datestr2dt(time_start),_datestr2dt(time_end),freq=str(interval)+'min')      
    datai_v = data['value']
    datai_v = np.interp(data_dates,data['date_time'],data['value'])
    datai = pd.DataFrame({'date_time':data_dates,'value':datai_v})
    data = datai
    
    # Determine the data points to get images for based on the value and cutoff arguements #
    if value=='all':
        # Get images for all of the observations #
        datas = data.dropna()
    elif ~isinstance(value,str):
        # Get cutoff images for observations equal to val (within a tolerance) #
        tol = 0.01
        datas = data
        datas = datas[datas['value']>=value-tol]
        datas = datas[datas['value']<=value+tol]
        datas = datas.sample(frac=1) # Shuffle the rows so various times are returned #
        datas = datas.iloc[0:cutoff]
        
    # Download image for each data point #
    datas['image'] = 0
    for i in range(len(datas)):
        time_start = _dt2datestr(datas['date_time'].iloc[i])
        time_end = _dt2datestr(datas['date_time'].iloc[i]+datetime.timedelta(minutes=1))
        filename = _call_WebcoosApi(camera=camera,
                                product=camera_product,
                                time_start=time_start,
                                time_end=time_end,
                                token=token,
                                save_dir=save_dir)
        if filename:
            datas['image'].iloc[i] = filename[0]
        else:
            datas['image'].iloc[i] = ''
            
    # If a view separator model has been provided, go through each image and label the view #
    if sep_model is not None:
        vs = ViewSeparator(camera,token,n_views=7)
        f = open(sep_model,'rb')
        modell = pickle.load(f)
        model = modell[0];decim_fac=modell[1]
        pred = []
        for i in range(len(datas)):
            imf = datas.iloc[i]['image']
            if len(imf)>0:
                im = plt.imread(imf)
                prediction = vs.predict(model,im,decim_fac=decim_fac)[0]
                pred.append(prediction)
            else:
                pred.append(np.nan)
        datas['view'] = pred
    else:
        datas['view'] = list(np.zeros(len(datas))*np.nan)
       
    return datas


def make_movie(datas,camera,station,view_num=None):
    '''
    Function to make a movie of synchronized data and webcamera imagery.
    
    Parameters
    _ _ _ _ _ 
    datas : Pandas DataFrame
        The synchroized data and webcamera frames returned by synch().
    camera : str
        The name of the camera from which datas was created.
    station : int
        The NWLON station ID from which datas was created.
    view_num : int, optional
        If a sep_model was given to synch(), the number of the view for which you want to create a movie. Otherwise None.
        Default is None.
    
    Returns
    _ _ _ _ _ 
    video_file : str
        The full path to the video file (.mp4) that is created.
    '''
    # Create and save frames #
    print('Making movie frames...')
    datas_mov = _save_frames(datas,camera,station,view_num)  
    
    # Make the movie #
    print('Producing the movie...')
    video_file = _produce_movie(datas_mov)
    return video_file

    

def _call_CoopsApi(station,data_product,time_start,time_end):
    '''
    Call the CO-OPS API to download requested data.
    '''
    coopsapi = CoopsApi()
    start_date = _datestr2dt(time_start)
    end_date = _datestr2dt(time_end)
    [data, data_errors] = coopsapi.get_data(str(station),[start_date,end_date],product=data_product,datum = 'MHHW',timeZone='lst_ldt')
    return data

def _get_flood_thresh(station):
    '''
    Get the NWS thresholds for the station relative to MHHW.
    '''
    lvls = CoopsApiTwo(station,'20240101','20240102',product='floodlevels').run()['floodlevels']
    datums = CoopsApiTwo(station,'20240101','20240102',product='datums').run()['datums']['datums'] # Datums are returned relative to the station datum #
    mhhw = np.array(datums)[np.array([datums[i]['name'] for i in range(len(datums))]) == 'MHHW'][0]['value']

    for k in lvls.keys():
        try:
            lvls[k] = lvls[k]-mhhw
        except:
            pass
    return lvls

def _call_WebcoosApi(camera,product,time_start,time_end,token,save_dir):
    '''
    Call the WebCOOS API to download images.
    '''
    api =  pywebcoos.API(token)
    filenames = api.download(camera_name=camera,
                             product_name=product,
                             start=time_start,
                             stop=time_end,
                             interval=1,
                             save_dir=save_dir)
    return filenames

def _datestr2dt(datestr):
    '''
    Convert a date-string of the form yyyymmddHHMM to a datetime.datetime object.
    '''
    dt = datetime.datetime(int(datestr[0:4]),
                           int(datestr[4:6]),
                           int(datestr[6:8]),
                           int(datestr[8:10]),
                           int(datestr[10:12]),
                           0)
    return dt

def _dt2datestr(dt):
    '''
    Convert a datetime.datetime object to a date-string of the form yyyymmddHHMM.
    '''
    mo = dt.month
    day = dt.day
    hr = dt.hour
    mn = dt.minute
    if mo<10:
        smo = '0'
    else:
        smo = ''
    if day<10:
        sday = '0'
    else:
        sday = ''
    if hr<10:
        shr = '0'
    else:
        shr = ''
    if mn<10:
        smn = '0'
    else:
        smn = ''
    return str(dt.year)+smo+str(mo)+sday+str(day)+shr+str(hr)+smn+str(mn)

def _save_frames(datas,camera,station,view_num):
    '''
    Save each frame of a movie.
    '''
    datas['frame'] = 0
    
    if datas['date_time'].diff().dropna().nunique()==1: # If datas is evenly spaced in time, then val='all' and we want a timeseries.
        ts=True
    else: # If datas is not evenly spaced in time, then user wants images at a certain value, and we want to show that value #
        ts=False 
        
    lens = np.array([len(datas['image'].iloc[i]) for i in range(len(datas))])
    datas = datas[lens>0]                     
                     
    if view_num is not None:
        ii = np.where(datas['view']==view_num)[0]
    else:
        ii = range(len(datas))
               
    for i in ii:
        if ts:
            # Get the timeseries frame template #
            fig,ax_img,ax_data = _FrameTemplates().ts(datas)            
            # Plot previous data as a line and the current data as a point #
            line = ax_data.plot(datas['date_time'].iloc[0:i+1].dt.to_pydatetime(),datas['value'].iloc[0:i+1],color='cornflowerblue')
            scat = ax_data.scatter(datas['date_time'].iloc[i].to_pydatetime(),datas['value'].iloc[i],150,color='cornflowerblue',edgecolor='darkgray',linewidth=2,alpha=1)
        else:
            # Get the bar frame template #
            fig,ax_img,ax_data = _FrameTemplates().bar(datas,station)
            # Plot the current data as a bar #
            ax_data.bar(1,datas['value'].iloc[i],width=0.5,color='cornflowerblue',edgecolor='darkgray')
            
        # Do some type agnostic axis formatting #
        ax_img.set_xticklabels([])
        ax_img.set_yticklabels([])
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_data.grid('on')
        ax_data.set_ylabel('Height in meters (MHHW)')
        ax_img.set_title('Camera: '+camera+'\nGauge: '+str(station),fontsize=10,fontweight='normal',loc='left')

        # Plot the NWS flood threshold levels on the data axis #
        mhhw = 0
        if ax_data.get_ylim()[0]<mhhw<ax_data.get_ylim()[1]: # Plot the mhhw level if the data covers it #
            ax_data.plot(ax_data.get_xlim(),[mhhw,mhhw],'k--')
            ax_data.text(ax_data.get_xlim()[0],mhhw,'MHHW',color='k',fontsize=12)
        minor = _get_flood_thresh(station)['nws_minor']
        if ax_data.get_ylim()[0]<minor<ax_data.get_ylim()[1]: # Plot the minor flood level if the data covers it #
            ax_data.plot(ax_data.get_xlim(),[minor,minor],'y--')
            ax_data.text(ax_data.get_xlim()[0],minor,'Minor flooding',color='y',fontsize=12)
        moderate = _get_flood_thresh(station)['nws_moderate']
        if ax_data.get_ylim()[0]<moderate<ax_data.get_ylim()[1]: # Plot the moderate flood level if the data covers it #
            ax_data.plot(ax_data.get_xlim(),[moderate,moderate],'r--')
            ax_data.text(ax_data.get_xlim()[0],moderate,'Moderate flooding',color='r',fontsize=12)        
        major = _get_flood_thresh(station)['nws_major']
        if ax_data.get_ylim()[0]<major<ax_data.get_ylim()[1]: # Plot the major flood level if the data covers it #
            ax_data.plot(ax_data.get_xlim(),[major,major],'m--')
            ax_data.text(ax_data.get_xlim()[0],major,'Major flooding',color='m',fontsize=12)

        # Plot image on the image axis #          
        im = plt.imread(datas['image'].iloc[i])
        ax_img.imshow(im)

        # Save the frame #
        savename = datas['image'].iloc[i]+'_frame.png'            
        plt.savefig(savename,dpi=450)
        datas['frame'].iloc[i] = savename
        plt.close()
        
    return datas

def _produce_movie(datas_mov):
    '''
    Create a movie from saved frames.
    '''
    # Determine desired movie fps #
    if datas_mov['date_time'].diff().dropna().nunique()==1: # If datas is evenly spaced in time, then val='all' and we want a timeseries, fps should be fast
        fps=4
    else: # If datas is not evenly spaced in time, then user wants images at a certain value, an the movie should be slower #
        fps=2
        
    # If a sep_model was used, frames were only saved for images with the desired view_num.
    # Need to drop entries without frames, which have values = 0 #
    datas_mov = datas_mov[~(datas_mov['frame']==0)]
        
    # Gather the frames #
    clips = []
    for i in range(len(datas_mov)):
        frame_path = datas_mov['frame'].iloc[i]
        clip = ImageSequenceClip([frame_path], fps=fps).set_duration(1/fps)            
        clips.append(clip)

    # Concatenate all clips into one video
    final_clip = concatenate_videoclips(clips)
    
    #Write the clip
    lens = np.array([len(datas_mov['image'].iloc[i]) for i in range(len(datas_mov))])
    good = np.where(lens>0)[0]
    if len(good)>0:
        out_file_name = os.path.basename(datas_mov['image'].iloc[good[0]]).split('.')[0]+'--'+os.path.basename(datas_mov['image'].iloc[good[-1]]).split('.')[0]+'--'+'Video.mp4'
        out_file = os.path.join(os.path.dirname(datas_mov['image'].iloc[good[0]]),out_file_name)
    else:
        out_file = 'Video.mp4' ### TODO ###                                                                               
    final_clip.write_videofile(out_file, codec='libx264')
    return out_file



class _FrameTemplates():
    '''
    Class that provides movie frame templates as methods.
    Currenly supported templates are ts (timeseries) and bar (bar plot).
    '''
    @staticmethod
    def ts(datas):
        fig = plt.figure(figsize=(9,8)) 
        ax_img = plt.subplot2grid((20, 19), (0, 0),colspan=19,rowspan=11)
        ax_data = plt.subplot2grid((20, 19), (12, 0), colspan=19,rowspan=9)
        plt.subplots_adjust(hspace=0)
        ax_data.set_xlim(datas['date_time'].iloc[0].to_pydatetime(),datas['date_time'].iloc[-1].to_pydatetime())
        ax_data.set_ylim(np.min(datas['value'])-0.1,np.max(datas['value'])+0.1)
        ax_data.xaxis.set_major_formatter(DateFormatter('%m/%d/%y\n%H:%M'))      
        return fig,ax_img,ax_data
    
    @staticmethod
    def bar(datas,station):
        fig = plt.figure(figsize=(9,4)) 
        ax_img = plt.subplot2grid((11, 19), (0, 0),colspan=12,rowspan=11)
        ax_data = plt.subplot2grid((11, 19), (1, 13), colspan=6,rowspan=9)
        ax_data.yaxis.tick_right()
        ax_data.yaxis.set_label_position("right")
        plt.subplots_adjust(wspace=0)
        ax_data.set_ylim(min([np.min(datas['value']),0])-0.05,_get_flood_thresh(station)['nws_major']+0.1)
        ax_data.set_xlim(0.5,1.5)
        ax_data.set_xticks([1])
        ax_data.set_xticklabels([])
        return fig,ax_img,ax_data
    
        
        
class ViewSeparator():
    """
    A class that represents a tool to separate views from a PTZ (rotating) camera. The tool is first trained on data and can then be applied.

    Attributes:
    -----------
    camera : str
        The name of the camera from which datas was created.
    token : str
        WebCOOS API token.
    n_views : int
        The number of unique views through which the camera rotates.


    Methods:
    --------
    get_random_images(time_start,time_stop,n,direc):
        Saves n random images between time_start and time_stop to direc, to use for ML model training or testing.
    label(direc):
        Interactively allows a user to assign labels to saved images.
    train(direc,decim_fac):
        Trains a convolutional neural network on labelled images.
    predict(model,direc_or_image,decim_fac=10):
        Predicts labels for a directory of images or single image given a trained model.
    inspect_prediction(direc,prediction):
        Interactively shows the user the predicted labels of images in direc.
    save_model(model,decim_fac,direc,fname):
        Saves necessary model parameters for loading later.

    Example:
    --------
    vs = ViewSeparator('Charleston Harbor, SC',token,n_views=7)
    vs.get_random_images('202401010000','202410012359',n=250,direc='train_direc')
    vs.label('train_direc')
    model = vs.train('train_direc',decim_fac=10)
    vs.save_model(model,decim_fac=10,'save_direc','trained_model')
    """
    def __init__(self,camera,token,n_views):
        self.camera = camera
        self.token = token
        self.n_views = n_views
        
    def get_random_images(self,time_start,time_stop,n,direc):
        times = self._get_random_images_times(time_start,time_stop,n)
        for i in range(len(times)):
            time_start = _dt2datestr(times[i])
            time_end = _dt2datestr(times[i]+datetime.timedelta(minutes=1))
            filename = _call_WebcoosApi(camera=self.camera,
                                    product='one-minute-stills',
                                    time_start=time_start,
                                    time_end=time_end,
                                    token=self.token,
                                    save_dir=direc)
    
    def label(self,direc):
        ims = os.listdir(direc) 
        for imf in ims:
            imff = os.path.join(direc,imf)
            try:
                im = plt.imread(imff)
            except:
                pass
            else:
                if not os.path.exists(imff.split('.')[0]+'_label'+'.pkl'): # Skip if the image already has a label file #
                    # Show the image
                    plt.imshow(im)   
                    plt.axis('off')  # Hide axes
                    plt.show()
                    # Ask for user input
                    user_input = input("You specified that this camera has "+str(self.n_views)+" views. Enter the number of this view  (note that entering 0 will place this image in the 'other' category). ")
                    print("Your input:", user_input)
                    label = int(user_input)
                    # Save the label alongside the image #
                    with open(imff.split('.')[0]+'_label'+'.pkl','wb') as f:
                        pickle.dump(label,f)
                    plt.close('all')
    
    def train(self,direc,decim_fac=10):
        dset = self._make_dataset(direc,decim_fac)
        model = self._do_train(dset)
        return model
    
    def predict(self,model,direc_or_image,decim_fac=10):
        dset_pred = self._make_dataset(direc_or_image,decim_fac)
        prediction = self._do_predict(model,dset_pred)
        return prediction     
    
    def inspect_prediction(self,direc,prediction):
        files = glob.glob(direc+'/*.jpg')
        c=-1
        for f in files:
            c+=1
            im = plt.imread(f)
            plt.imshow(im)    
            plt.title('Predicted label = '+str(prediction[c]))
            plt.show()

            # Wait for user input to continue
            input("Press Enter to continue to the next image...")    
            # Clear the output for the next image
            plt.close('all')
            
    def save_model(self,model,decim_fac,direc,fname):
        with open(direc+'/'+fname+'.pkl','wb') as f:
            pickle.dump([model,decim_fac],f)

        
    @staticmethod
    def _get_random_images_times(time_start,time_stop,n):
        dt_start = _datestr2dt(time_start)
        dt_stop = _datestr2dt(time_stop)
        r = pd.date_range(dt_start,dt_stop,freq='min')
        i_take = [random.randint(0,len(r)) for _ in range(n)]
        return r[i_take].to_pydatetime()
    
    @staticmethod
    def _make_dataset(direc_or_image,decim_fac):
        if isinstance(direc_or_image,str):
            files = glob.glob(direc_or_image+'/*.jpg')
            dset = pd.DataFrame({'label':[],'image':[]})
            for f in files:
                ims = f
                labs = f.split('.')[0]+'_label.pkl'
                im = plt.imread(ims)
                try:
                    f = open(labs,'rb');lab = pickle.load(f)
                except:
                    lab = 1000
                dset = pd.concat([dset,pd.DataFrame({'label':int(lab),'image':[im]},index=[0])],ignore_index=True)
            dset['image vec'] = dset.apply(lambda row: np.array(row.image).flatten()[0:-1:decim_fac] , axis = 1).tolist()
        else:
            dset = pd.DataFrame({'image':[direc_or_image]})
            dset['image vec'] = dset.apply(lambda row: np.array(row.image).flatten()[0:-1:decim_fac] , axis = 1).tolist()
        return dset
    
    @staticmethod
    def _do_train(dset):
        X = np.vstack(dset['image vec'].tolist()) / 255.0
        y = dset['label']
        mlp_model = MLPClassifier(hidden_layer_sizes=[300,100], 
                          activation='relu', 
                          early_stopping=True,
                          random_state=13, 
                          verbose= True)

        mlp_model.fit(X,y)
        return mlp_model
    
    @staticmethod
    def _do_predict(model,dset_pred):
        X = np.vstack(dset_pred['image vec'].tolist()) / 255.0
        pred = model.predict(X)
        return pred
    
