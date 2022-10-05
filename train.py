# =========== Import pytorch libs ===========
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as Data
# =========== Import Option modules ===========
from options.train_options import TrainOptions
# =========== Import Dataset modules ===========
import data
# =========== Import Models ===========
from models import create_model
from trainers.kpe_trainer import KPETrainer
# =========== Import Training tools ===========
# =========== Import util modules ===========
from util.visualizer import Visualizer
from util.iter_counter import IterationCounter
# =========== Import etc... ===========
import sys

if __name__ == "__main__":
    # parse options
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))

    dataloader = data.create_dataloader(opt)

    # create trainer for our model
    trainer = KPETrainer(opt)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))
    # create tool for visualization
    visualizer = Visualizer(opt)
    # cuda는 model안에서 바꿔주는걸로
    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            trainer.run_generator_one_step(data_i)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            # if iter_counter.needs_displaying():
                # visuals = OrderedDict([('input_label', data_i['label']),
                #                        ('synthesized_image', trainer.get_latest_generated()),
                #                        ('real_image', data_i['image'])])
                # visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()
        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
                epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    print('Training was successfully finished.')

    #
    # h_grid_size = 2 / opt.h_grid # (1 - (-1)) / opt.h_grid
    # w_grid_size = 2 / opt.w_grid # (1 - (-1)) / opt.w_grid
    #
    # # data_path = 'dataset/train/pose_label.pkl'
    # print('Make Batch Dataset', end = '...')
    # train_df = utils.load_train_data('dataset/train_annotation.csv')
    # src, tgt_with_occlusion, mid_point, length = Make_batch(train_df, opt).get_batch()
    # train_data = MyDataSet(src, tgt_with_occlusion, mid_point, length)
    #
    # valid_df = utils.load_train_data('dataset/valid_annotation.csv')
    # src, tgt_with_occlusion, mid_point, length = Make_batch(valid_df, opt).get_batch()
    # valid_data = MyDataSet(src, tgt_with_occlusion, mid_point, length)
    #
    # print('Done!!', end = '...')
    # train_dl = Data.DataLoader(train_data, opt.batch_size, True)
    # valid_dl = Data.DataLoader(valid_data, opt.batch_size, True)
    #
    # grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])
    # opt.grid_size_tensor = grid_size_tensor.to(opt.device)
    #
    # model = create_model(opt)
    # model_optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    # scheduler = ReduceLROnPlateau(model_optimizer, 'min', verbose=True)
    #
    # model, model_optimizer, scheduler, epoch, loss = utils.load_model(opt, model, model_optimizer, scheduler)
    # opt.epoch = epoch
    # opt.loss = loss
    # # opt.epoch = 1
    # # opt.loss = 0
    # trainer = Trainer(opt=opt,
    #                   model=model)
    #
    # trainer.trainIters(opt=opt,
    #                    model_optimizer=model_optimizer,
    #                    scheduler=scheduler,
    #                    train_dl=train_dl,
    #                    valid_dl=valid_dl)



