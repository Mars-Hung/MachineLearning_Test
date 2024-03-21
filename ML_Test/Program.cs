using Microsoft.ML;
using Microsoft.ML.Data;
// See https://aka.ms/new-console-template for more information

Console.WriteLine("Would you like to re-train the model? (yes/no)");
string response = Console.ReadLine();
if (response.ToLower() == "yes")
{
    TrainModel();
}
else if (response.ToLower() == "no")
{
}
else
{
    Console.WriteLine("Invalid response. Please enter 'yes' or 'no'.");
    return;
}
do
{


    Console.WriteLine("Please enter your name:");
    string name = Console.ReadLine();

    PersonPrediction prediction = PredictGender(name);
    Console.WriteLine($"Predicted Gender: {(prediction.Prediction ? "Male" : "Female")}");
    Console.WriteLine("");
    Console.WriteLine("");

} while (true);



static PersonPrediction PredictGender(string name)
{
    // 初始化ML.NET的机器学习上下文
    var mlContext = new MLContext();

    // 加载模型
    ITransformer loadedModel = mlContext.Model.Load("model.zip", out var schema);

    // 创建预测引擎
    var predictionEngine = mlContext.Model.CreatePredictionEngine<PersonData, PersonPrediction>(loadedModel);

    // 进行预测
    var prediction = predictionEngine.Predict(new PersonData { Name = name });

    return prediction;
}
// 训练模型
static void TrainModel()
{
    Console.WriteLine("Please wait...");
    // 初始化ML.NET的机器学习上下文
    var mlContext = new MLContext();

    // 加载数据集
    IDataView dataView = mlContext.Data.LoadFromTextFile<PersonData>("people_data.csv", separatorChar: ',');

    // 数据预处理和特征工程
    var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("NameFeaturized", "Name");

    // 选择模型
    var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "NameFeaturized");

    // 构建机器学习管道
    var trainingPipeline = dataProcessPipeline.Append(trainer);

    // 训练模型
    var model = trainingPipeline.Fit(dataView);

    // 保存模型
    mlContext.Model.Save(model, dataView.Schema, "model.zip");

    Console.WriteLine("Model trained and saved successfully.");
}
static ITransformer LoadModel()
{
    try
    {
        var mlContext = new MLContext();
        return mlContext.Model.Load("model.zip", out var schema);
    }
    catch (FileNotFoundException)
    {
        return null; // 没有可用的模型
    }
}
// 定义数据模型类
public class PersonData
{
    [LoadColumn(0)]
    public string Name { get; set; }

    [LoadColumn(1), ColumnName("Label")]
    public bool Gender { get; set; }
}

// 定义预测结果类
public class PersonPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}
