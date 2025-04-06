def print_log(lr, epoch, max_epoch, iter_i, epoch_size, loss_dict, time, accumulate):
    # basic infor
    log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
    log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
    log += '[lr: {:.6f}]'.format(lr[0])
    # loss infor
    for k in loss_dict.keys():

        if k == 'losses':
            log += '[{}: {:.2f}]'.format(k, loss_dict[k] * accumulate)
        else:
            log += '[{}: {:.2f}]'.format(k, loss_dict[k])

    # other infor
    log += '[time: {:.2f}]'.format(time)

    # print log infor
    print(log, flush=True)