# -*- coding: utf-8 -*-
"""
Script Name: Training Log Analysis and Visualization
Created on: Wed Feb 7 12:35:14 2024

Author: Laura Elena Cue La Rosa
Project: WUR-WWF Deforestation Project

Description:
    This script parses training logs generated during the training of deep learning models on deforestation analysis tasks.
    It extracts epoch-wise performance metrics such as loss, accuracy, precision, recall, F1 score, and learning thresholds
    from the logs. These metrics are plotted to visualize the model's training progression and testing performance.

    The script supports handling multiple experimental configurations stored under a specified directory. It assumes
    logs are formatted in a specific regex pattern, which is crucial for correct parsing.

Usage:
    Adjust 'main_path' and 'path_model' to point to the directory containing the experiment logs and specify the model
    directories respectively. Run the script from a Python environment where all dependencies (matplotlib, pandas, os, re) are installed.

    The script outputs plots directly into the experiment directories, providing a visual summary of each training run.

"""

import re
import matplotlib.pyplot as plt
import os

import pandas as pd
main_path = "./exp_tiles_continent_v2/Africa/"

path_model = ["resunet"]


pattern = re.compile(
    r"Epoch: \[(\d+)\]\[200\]\s+Loss (\d+\.\d+) \((\d+\.\d+)\)\s+Acc (\d+\.\d+) \((\d+\.\d+)\)\s+F0.5 (\d+\.\d+) \((\d+\.\d+)\)\s+Pre (\d+\.\d+) \((\d+\.\d+)\)\s+Rec (\d+\.\d+) \((\d+\.\d+)\)\s+Tresh (\d+\.\d+(?:e[+-]?\d+)?) \((\d+\.\d+(?:e[+-]?\d+)?)\)\s+Lr: (\d+\.\d+)"
)

test_pattern = re.compile(
    r"Tresh \((\d+\.\d+)\)\s+Pre positive \((\d+\.\d+)\)\s+Rec positive \((\d+\.\d+)\)\s+F0.5 \((\d+\.\d+)\)\s+Acc \((\d+\.\d+)\)"
)


for pm in path_model:
    
    exp_paths = [d for d in os.listdir(os.path.join(main_path,pm)) if os.path.isdir(os.path.join(main_path,pm,d))]
    
    for ep in exp_paths:
        exp_path = os.path.join(main_path,pm,ep)
        
        # Initialize variables to store extracted data
        epochs = []
        train_f1 = []
        train_pre = []
        train_rec = []
        train_losses_last_iter = []
        tresh = []
        
        test_f1 = []
        test_pre = []
        test_rec = []
        
        tresh_test = []
        try:
          with open(os.path.join(exp_path,"train.log"), "r") as file:
              for line in file:
                  
                  match = pattern.search(line)
                  if match:
                      epoch, _, loss_paren, _, acc_paren, _, f1_paren, _, pre_paren, _, rec_paren, _, tresh_paren, lr = match.groups()
                  
                      epochs.append(epoch)
                      
                      train_losses_last_iter.append(float(loss_paren))
                      
                      train_f1.append(float(f1_paren))
                      
                      train_pre.append(float(pre_paren))
                      
                      train_rec.append(float(rec_paren))
                          
                      tresh.append(float(tresh_paren))
                      
                  match_test = test_pattern.search(line)
                  if match_test:
                      tresh_te, pre_positive, rec_positive, f1_paren, acc = match_test.groups()
                      
                      test_f1.append(float(f1_paren))
                      
                      test_pre.append(float(pre_positive))
                      
                      test_rec.append(float(rec_positive))
                          
                      tresh_test.append(float(tresh_te))
                      
          
          # Ensure all lists have the same length for plotting
          min_length = min(len(train_f1), len(test_f1))
          epochs = epochs[:min_length]
          train_losses_last_iter = train_losses_last_iter[:min_length]
          train_f1 = train_f1[:min_length]
          train_pre = train_pre[:min_length]
          train_rec = train_rec[:min_length]
          tresh = tresh[:min_length]
          
          test_f1 = test_f1[:min_length]
          test_pre = test_pre[:min_length]
          test_rec = test_rec[:min_length]
          tresh_test = tresh_test[:min_length]
          
          # Plotting
          fig, ax1 = plt.subplots(figsize=(10, 6))
          
          # Set xlabel for ax1
          ax1.set_xlabel('Epoch')
          ax1.set_ylabel('Training Loss', color='black')
          
          # Plot Training Loss
          color_training_loss = 'tab:red'
          ax1.plot(epochs, train_losses_last_iter, color=color_training_loss, linestyle='--', label='Training Loss')
          ax1.tick_params(axis='y', labelcolor=color_training_loss)
          
          # Plot Training Tresh
          color_training_tresh = 'tab:purple'  # Changed color to make it distinct
          ax1.plot(epochs, tresh, color=color_training_tresh, linestyle='--', label='Training Tresh')
          
          # Plot Test Tresh with a different color
          color_test_tresh = 'tab:olive'  # Changed color to make it distinct
          ax1.plot(epochs, tresh_test, color=color_test_tresh, linestyle='--', label='Test Tresh')
          
          # Ensure the ylabel and ticks are set correctly for ax1
          ax1.tick_params(axis='y', labelcolor='black')
          
          # Create a twin Axes sharing the x-axis for the other metrics
          ax2 = ax1.twinx()
          ax2.set_ylabel('Metrics', color='black')  # Combined label for clarity
          
          # Plot Training and Testing F1, Pre, Rec with distinct colors
          color_train_f1 = 'tab:blue'
          ax2.plot(epochs, train_f1, color=color_train_f1, linestyle='-', label='Training F1')
          
          color_train_pre = 'tab:orange'
          ax2.plot(epochs, train_pre, color=color_train_pre, linestyle='-', label='Training Pre')
          
          color_train_rec = 'tab:green'
          ax2.plot(epochs, train_rec, color=color_train_rec, linestyle='-', label='Training Rec')
          
          color_test_f1 = 'tab:cyan'  # Use a different color for test metrics
          ax2.plot(epochs, test_f1, color=color_test_f1, linestyle='-', label='Test F1')
          
          color_test_pre = 'tab:pink'  # Use a different color for test metrics
          ax2.plot(epochs, test_pre, color=color_test_pre, linestyle='-', label='Test Pre')
          
          color_test_rec = 'tab:brown'  # Use a different color for test metrics
          ax2.plot(epochs, test_rec, color=color_test_rec, linestyle='-', label='Test Rec')
          
          # Annotate the last values with corresponding colors
          ax2.text(epochs[-1], train_f1[-1], f"{train_f1[-1]:.2f}", color=color_train_f1, va='bottom')
          ax2.text(epochs[-1], train_pre[-1], f"{train_pre[-1]:.2f}", color=color_train_pre, va='bottom')
          ax2.text(epochs[-1], train_rec[-1], f"{train_rec[-1]:.2f}", color=color_train_rec, va='bottom')
          
          ax2.text(epochs[-1], test_f1[-1], f"{test_f1[-1]:.2f}", color=color_test_f1, va='bottom')
          ax2.text(epochs[-1], test_pre[-1], f"{test_pre[-1]:.2f}", color=color_test_pre, va='bottom')
          ax2.text(epochs[-1], test_rec[-1], f"{test_rec[-1]:.2f}", color=color_test_rec, va='bottom')
  
  
          
          # Title and grid
          plt.title('Training Loss and Metrics per Epoch')
          fig.tight_layout()
          
          # Adding legends
          # Need to handle legends when having lines on both ax1 and ax2
          lines1, labels1 = ax1.get_legend_handles_labels()
          lines2, labels2 = ax2.get_legend_handles_labels()
          ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
          
          # Set only the first and last x-axis tick labels
          plt.gca().set_xticks([min(epochs), max(epochs)])  # Set ticks at the min and max of x
          plt.gca().set_xticklabels([f'{min(epochs)}', f'{max(epochs)}'])  # Set tick labels as strings of min and max of x
  
          
          # Save the figure
          plt.savefig(os.path.join(exp_path,'training_loss_plot.png'), dpi=300, format='png', bbox_inches='tight')
          plt.close()
          plt.cla()
        except:
          print('No data inside folder')
        
        