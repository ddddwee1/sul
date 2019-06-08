		# === Read and denormalize the gt with srnn's seeds, as we'll need them
		# many times for evaluation in Euler Angles ===
		srnn_gts_expmap = get_srnn_gts( actions, model, test_set, data_mean,
															data_std, dim_to_ignore, not FLAGS.omit_one_hot, to_euler=False )
		srnn_gts_euler = get_srnn_gts( actions, model, test_set, data_mean,
															data_std, dim_to_ignore, not FLAGS.omit_one_hot )
		print(srnn_gts_euler['walking'][0])
		input()

		avg_mean_errors = np.zeros([model.target_seq_len])
		# Predict and save for each action
		for action in actions:

			# Make prediction with srnn' seeds
			encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn( test_set, action, actions )
			forward_only = True
			srnn_seeds = True
			srnn_loss, srnn_poses, _, = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, forward_only, srnn_seeds)

			# denormalizes too
			srnn_pred_expmap = data_utils.revert_output_format( srnn_poses, data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )

			# Save the conditioning seeds


			# Compute and save the errors here
			mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

			for i in np.arange(8):

				eulerchannels_pred = srnn_pred_expmap[i]

				for j in np.arange( eulerchannels_pred.shape[0] ):
					for k in np.arange(3,97,3):
						eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
							data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

				eulerchannels_pred[:,0:6] = 0

				# Pick only the dimensions with sufficient standard deviation. Others are ignored.
				idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

				euc_error = np.power( srnn_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
				euc_error = np.sum(euc_error, 1)
				euc_error = np.sqrt( euc_error )
				print('sequence: ', i, 'error: ', euc_error)
				mean_errors[i,:] = euc_error

			mean_mean_errors = np.mean( mean_errors, 0 )
			print( action )
			print( ','.join(map(str, mean_mean_errors.tolist() )) )

			print( 'Subset results for 80ms, 160ms, 320ms, 400ms' )
			print( ','.join(map(str, mean_mean_errors[[1, 3, 7, 9]].tolist() )) )
			if FLAGS.seq_length_out >= 25:
				print( 'Subset results for 80ms, 160ms, 320ms, 400ms, 560ms, 1000ms' )
				print( ','.join(map(str, mean_mean_errors[[1, 3, 7, 9, 13, 24]].tolist() )) )