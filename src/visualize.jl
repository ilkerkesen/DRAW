function mnistgrid(images, nrow, filename=nothing)
    yy = reshape(images, 28, 28, 1, size(images)[end])
    yy = permutedims(yy, (2,1,3,4))
    yy = Images.imresize(yy, 40, 40)
    yy = Gray.(mosaicview(yy, 0.5f0, nrow=nrow, npad=1, rowmajor=true))
    if filename == nothing
        display(yy)
    else
        save(filename, yy)
    end
end
