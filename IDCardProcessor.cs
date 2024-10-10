using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Tesseract;
using ZXing;
using Emgu.CV;
using System;
using System.Drawing;

public static class IDCardProcessorFunction
{
    [FunctionName("ProcessIDCard")]
    public static async Task<IActionResult> Run(
        [HttpTrigger(AuthorizationLevel.Function, "post", Route = "ProcessIDCard")] HttpRequest req,
        ILogger log)
    {
        log.LogInformation("Processing ID Card");

        var form = await req.ReadFormAsync();
        var file = form.Files["image"];

        if (file == null || file.Length == 0)
        {
            return new BadRequestObjectResult("Please upload an image file.");
        }

        string name = string.Empty;
        string barcode = string.Empty;
        string imageBase64 = string.Empty;

        try
        {
            var tempFilePath = Path.GetTempFileName();
            using (var stream = new FileStream(tempFilePath, FileMode.Create))
            {
                await file.CopyToAsync(stream);
            }

            name = ExtractName(tempFilePath);
            barcode = ExtractBarcode(tempFilePath);
            imageBase64 = ExtractPersonalImageUsingFaceDetection(tempFilePath);

            var result = new
            {
                Name = name,
                Barcode = barcode,
                ImageBase64 = imageBase64
            };

            return new OkObjectResult(result);
        }
        catch (Exception ex)
        {
            log.LogError($"Error processing ID card: {ex.Message}");
            return new StatusCodeResult(500);
        }
    }

    public static string ExtractName(string imagePath)
    {
        try
        {
            using (var ocrEngine = new TesseractEngine(@"C:\Program Files\Tesseract-OCR\tessdata", "eng", EngineMode.Default))
            {
                using (var img = Pix.LoadFromFile(imagePath))
                {
                    using (var page = ocrEngine.Process(img))
                    {
                        string extractedText = page.GetText();
                        return ExtractNameFromText(extractedText);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error in OCR: " + ex.Message);
            return "Name not found";
        }
    }

    public static string ExtractNameFromText(string text)
    {
        string[] lines = text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        if (lines.Length >= 4)
        {
            string firstName = lines[2].Trim();
            string lastName = lines[3].Trim();
            return firstName + " " + lastName;
        }
        return "Name not found";
    }

    public static string ExtractBarcode(string imagePath)
    {
        try
        {
            Bitmap barcodeBitmap = new Bitmap(imagePath);
            BarcodeReader barcodeReader = new BarcodeReader();
            var barcodeResult = barcodeReader.Decode(barcodeBitmap);

            if (barcodeResult != null)
            {
                return barcodeResult.Text;
            }
            else
            {
                return "Barcode not found";
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error in barcode extraction: " + ex.Message);
            return "Barcode not found";
        }
    }

    public static string ExtractPersonalImageUsingFaceDetection(string imagePath)
    {
        try
        {
            Mat image = CvInvoke.Imread(imagePath, Emgu.CV.CvEnum.ImreadModes.Color);
            string cascadeFilePath = Path.Combine(Directory.GetCurrentDirectory(), "Resources", "haarcascade_frontalface_default.xml");
            CascadeClassifier faceCascade = new CascadeClassifier(cascadeFilePath); Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            var faces = faceCascade.DetectMultiScale(grayImage, 1.1, 10, System.Drawing.Size.Empty);

            if (faces.Length > 0)
            {
                var faceRect = faces[0];
                int buffer = 42;
                var expandedRect = new Rectangle(
                    Math.Max(faceRect.X - buffer, 0),
                    Math.Max(faceRect.Y - buffer, 0),
                    Math.Min(faceRect.Width + 2 * buffer, image.Width - faceRect.X + buffer),
                    Math.Min(faceRect.Height + 2 * buffer, image.Height - faceRect.Y + buffer)
                );
                Mat personalImage = new Mat(image, expandedRect);

                using (var ms = new MemoryStream())
                {
                    string tempFilePath = Path.ChangeExtension(Path.GetTempFileName(), ".png");
                    CvInvoke.Imwrite(tempFilePath, personalImage);
                    using (var fileStream = new FileStream(tempFilePath, FileMode.Open, FileAccess.Read))
                    {
                        fileStream.CopyTo(ms);
                    }
                    byte[] imageBytes = ms.ToArray();
                    //commented code converts base64 back to image
                    // string base64String = Convert.ToBase64String(imageBytes);

                    // Image res_image = ConvertBase64ToImage(base64String);
                    // res_image.Save("output_image.png", System.Drawing.Imaging.ImageFormat.Png);
                    return Convert.ToBase64String(imageBytes);
                }
            }
            else
            {
                return "No face detected";
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error detecting face: " + ex.Message);
            return "Face detection failed";
        }
    }

    public static Image ConvertBase64ToImage(string base64String)
    {
        if (base64String.Contains(","))
        {
            base64String = base64String.Substring(base64String.IndexOf(",") + 1);
        }

        byte[] imageBytes = Convert.FromBase64String(base64String);

        using (var ms = new MemoryStream(imageBytes))
        {
            return Image.FromStream(ms);
        }
    }
}
