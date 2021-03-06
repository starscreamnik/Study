using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;

namespace AsyncImageFilter
{
    public class MatrixFilter
    {
        private static int _radius; // R - radius
        private readonly double[,] _data;
        public MatrixFilter(int dim)
        {
            var div = 0;
            _radius = (dim - 1) / 2; // k = 2r+1
            _data = new double[dim,dim];
            for (int i = 0, iVal = 1; i < dim; i++)
            {
                for (int j = 0, jVal = iVal; j < dim; j++)
                {
                    _data[i,j] = jVal;
                    div += jVal;

                    if (j < dim / 2) jVal *= 2;
                    else jVal /= 2;
                }

                if (i < dim / 2) iVal *= 2;
                else iVal /= 2;
            }

            for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                _data[i, j] /= div;

            Console.WriteLine("Matrix is completely created.");
        }

        private unsafe void SetPixel(byte* outLine, byte* src, int x, int y, int widthInBytes, int height, int bpp, int stride)
        {
            byte r = 0, g = 0, b = 0;
            for (var i = y - _radius; i <= y + _radius; i++)
            {
                var n = i < 0 ? 0 : i > height - 1 ? height - 1 : i;    // n - pixel row position       
               
                for (var j = x - _radius*bpp; j <= x + _radius*bpp; j+=bpp)
                {
                    var m = j < 0 ? 0 : j > widthInBytes - bpp ? widthInBytes - bpp : j; // m - byte column position                   
                    var pos = m + n * stride;    // byte position from first source image pixel
                    var pxl = Color.FromArgb(255,src[pos+2], src[pos+1], src[pos]);
                    r += (byte) (pxl.R * _data[i - y + _radius, (j-x)/bpp + _radius]);
                    g += (byte) (pxl.G * _data[i - y + _radius, (j-x)/bpp + _radius]);
                    b += (byte) (pxl.B * _data[i - y + _radius, (j-x)/bpp + _radius]);
                }
            }
            outLine[x] = b;
            outLine[x + 1] = g;
            outLine[x + 2] = r;
        }

        private unsafe void RunRow(byte* rowPtr, byte* src, int row, int byteWidth, int pixHeight, int bpp, int stride)
        {
            for (var x = 0; x < byteWidth; x += bpp)
                SetPixel(rowPtr, src, x, row, byteWidth, pixHeight, bpp, stride);
        }
        
        public unsafe Bitmap RunGaussianBlur(Bitmap srcImage, bool isAsync)
        {
            var doneImage = new Bitmap(srcImage.Width, srcImage.Height);

            var outData = doneImage.LockBits(
                new Rectangle(0, 0, doneImage.Width, doneImage.Height),
                ImageLockMode.ReadWrite, srcImage.PixelFormat
            );
            var srcData = srcImage.LockBits(
                new Rectangle(0, 0, srcImage.Width, srcImage.Height),
                ImageLockMode.ReadOnly, srcImage.PixelFormat
            );
            
            var bytesPerPixel = Image.GetPixelFormatSize(srcImage.PixelFormat) / 8;
            var heightInPixels = outData.Height;
            var widthInBytes = outData.Width * bytesPerPixel;
            var outFirstPixel = (byte*) outData.Scan0; // first pixel's address 
            var srcFirstPixel = (byte*) srcData.Scan0;

            if (isAsync)
            {
                var taskArray = new Task[heightInPixels];
                for (var i = 0; i < taskArray.Length; i++)
                {
                    var y = i;
                    byte* outCurrLine = outFirstPixel + y * outData.Stride;
                    taskArray[y] = Task.Run(
                        () => RunRow(outCurrLine, srcFirstPixel, y, widthInBytes,
                                     heightInPixels,bytesPerPixel, srcData.Stride)
                    );
                }

                Task.WaitAll(taskArray);
            }
            else
            {
                for (var y = 0; y < heightInPixels; y++)
                {
                    byte* outCurrLine = outFirstPixel + y * outData.Stride;
                    for (var x = 0; x < widthInBytes; x = x + bytesPerPixel)
                        SetPixel(outCurrLine, srcFirstPixel, x, y, widthInBytes, heightInPixels,bytesPerPixel, outData.Stride);
                }
            }

            doneImage.UnlockBits(outData);
            srcImage.UnlockBits(srcData);

            return doneImage;
        } 
    }
}