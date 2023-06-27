import argparse
import h5py
import sys
import logging
import cv2
import numpy as np
import os

def main():

    # ---- Script Arguments ---- #
    # Define 
    parser = argparse.ArgumentParser(description='Unpacks multipoint dataset into folder form usable by pix2pix (domain A = optical, domain B = thermal)')
    parser.add_argument('input_path',  help='Relative path to multipoint dataset')
    parser.add_argument('-o', dest='output_path', default='', help='Relative path to output folder. Default is current working directory.')
    parser.add_argument('-v', dest='validation', type=float, default=0.2, help='Portion of train set to use as validation (default=0.2)' )
    parser.add_argument('-r', dest='thermal_raw', action='store_true', help='Use raw thermal images')

    # Collect args
    args = parser.parse_args()

    # ---- Misc Script Setup ---- #
    # Set up logger 
    logger = logging.getLogger()

    # ---- Domains ---- #
    # NOTE: A = optical, B = thermal
    # TODO: fix hard-coding
    domains = {'A': 'optical', 'B': 'thermal' if not args.thermal_raw else 'thermal_raw'}
    
    # ---- Incoming Files ---- #
    # TODO: fix hard-coding
    #h5_filenames = {'test': 'test.hdf5', 'train': 'training.hdf5'} 
    dir_dataset_in  = os.path.join(os.getcwd(), args.input_path)

    # ---- Create Folders ---- #

    # Desired File Structure:
    # 
    #  multipoint/
    #   |--- A/
    #      |--- test/
    #         |--- 1.jpg/
    #      |--- train/
    #         |--- 2.jpg/
    #      |--- val/
    #         |--- 3.jpg/
    #   |--- B/
    #      |--- test/
    #         |--- 1.jpg/
    #      |--- train/
    #         |--- 2.jpg/
    #      |--- val/
    #         |--- 3.jpg/


    # ---- 1) Multipoint Directory ---- #
    dir_dataset_out = os.path.join(os.getcwd(), args.output_path)
    dir_dataset_out = os.path.join(dir_dataset_out, 'multipoint')

    # Throw error if directory exists
    try:
        os.mkdir(dir_dataset_out)
    except FileExistsError:
        logger.error('INFO: Multipoint directory already exists.')
        sys.exit()

    # Do the following for EACH domain to get the file structure:
    for domain_out, domain_in in domains.items(): 

        # ---- 2) Domain Directory ---- #
        dir_domain = os.path.join(dir_dataset_out, domain_out)
        os.mkdir(dir_domain)

        # ---- 3) Test Folder ---- #
        # Make the test folder
        dir_test = os.path.join(dir_domain, 'test')
        os.mkdir(dir_test)

        # Open test hdf5
        try:
            h5_test = h5py.File(os.path.join(dir_dataset_in, 'test.hdf5'), 'r')    
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, 'test.hdf5'))
            sys.exit()

        # Get image names
        memberslist = list(h5_test.keys())
        
        for member in memberslist:

            # Get image pair
            img_pair = h5_test[member]
            # Get proper domain
            img = img_pair[domain_in][:][:]
            # Convert to CV format
            img_cv = cv2.cvtColor((np.clip(img.squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
            # Save 
            cv2.imwrite(os.path.join(dir_test,member)+'.jpg', img_cv)

        # Close hdf5 file
        h5_test.close()

        # ---- 4) Train & Val Folder(s) ---- #
        # Make the train folder
        dir_train = os.path.join(dir_domain, 'train')
        os.mkdir(dir_train)

        # Make the val folder
        dir_val = os.path.join(dir_domain, 'val')
        os.mkdir(dir_val)

        # Open train hdf5
        try:
            h5_train = h5py.File(os.path.join(dir_dataset_in, 'training.hdf5'), 'r')    
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, 'training.hdf5'))
            sys.exit()

        # Get image names
        memberslist = list(h5_train.keys())
        
        # Split members into train & val
        num_train = int(len(memberslist)*(1-args.validation))
        memberslist_train = memberslist[0:num_train]
        memberslist_val   = memberslist[num_train:]

        for member in memberslist_train:

            # Get image pair
            img_pair = h5_train[member]
            # Get proper domain
            img = img_pair[domain_in][:][:]
            # Convert to CV format
            img_cv = cv2.cvtColor((np.clip(img.squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
            # Save 
            cv2.imwrite(os.path.join(dir_train,member)+'.jpg', img_cv)

        for member in memberslist_val:

            # Get image pair
            img_pair = h5_train[member]
            # Get proper domain
            img = img_pair[domain_in][:][:]
            # Convert to CV format
            img_cv = cv2.cvtColor((np.clip(img.squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
            # Save 
            cv2.imwrite(os.path.join(dir_val,member)+'.jpg', img_cv)

        # Close hdf5 file
        h5_train.close()



if __name__ == "__main__":
    main()
