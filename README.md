간략하게

rank는 개별 gpu 넘버 / world size는 전체 gpu 개수

multi-gpus로 학습하는 경우, 배치사이즈별로 나눠서 여러개 gpu로 학습할 수 있음

ex) 2개의 서버 / 각 서버는 4개의 gpu를 가진다 가정하면
총 8개의 gpu가 있고, world size는 따라서 8개

ex) batch size  = 32이고 world size가 8이면 
배치사이즈를 world size로 나눈 만큼 gpu 8개가 나눠서 학습
32/8=4 이므로 batch size=4인 (데이터 4개씩) 배치를 gpu 8개가 동시에 학습

rank는 8이고 rank는 [0,1,2,3,4,5,6,7]이 된다.

rank가 0이 되면 결국 한 epoch을 다 학습하였다고 볼 수 있고 모델파라미터를 저장한다. 아래 SSL code snippet(Barlow Twins)에서도 확인해볼 수 있다.


```python
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')
```

[DDP all reduce](https://blahblahlab.tistory.com/205)



