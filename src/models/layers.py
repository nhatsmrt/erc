#
#
# class ConvolutionalLayerWithDropout(nn.Sequential):
#     """
#     Simple convolutional layer: input -> conv2d -> activation -> norm 2d
#     """
#     def __init__(
#             self, in_channels, out_channels,
#             kernel_size=3, stride=1, padding=0,
#             bias=False, activation=nn.ReLU, normalization=nn.BatchNorm2d
#     ):
#         super().__init__()
#         self.add_module(
#             "main",
#             nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     stride=stride,
#                     padding=padding,
#                     bias=bias
#                 ),
#                 activation(),
#                 normalization(num_features=out_channels)
#             )
#         )
