% P2: audios en bruto + L2, cambio de la arquitectura de la red neuronal

%% Creo la matriz de etiquetas con las 8 emociones a partir de la base de datos RAVDESS

% Ruta de los archivos de audio de la base de datos RAVDESS
audios_ravdess = dir('C:\Users\maria.martinez\OneDrive - VINCI Energies\Documentos\Nueva carpeta\1440audios\*.wav');

% Creo una matriz vacía para almacenar las etiquetas
emotions_ravdess = zeros(length(audios_ravdess), 1);

% Creo una matriz para almacenar las etiquetas en formato one-hot porque si
% no en el entrenamiento me da problemas
numEmotions = 8; % Número total de emociones
emotions_onehot_ravdess = zeros(length(audios_ravdess), numEmotions);

for i = 1:length(audios_ravdess)
    filename = audios_ravdess(i).name;
    [audio, fs] = audioread(filename);
       
    % Obtengo la etiqueta del archivo de audio
    label = filename(7:8); % En este caso, el segundo y tercer dígito del nombre del archivo corresponden a la etiqueta
    
    % Asigno valores del 1-8 a las emociones
    switch label
        case '01'
            emotions_ravdess(i) = 1; % Neutral
        case '02'
            emotions_ravdess(i) = 2; % Calm
        case '03'
            emotions_ravdess(i) = 3; % Happy
        case '04'
            emotions_ravdess(i) = 4; % Sad
        case '05'
            emotions_ravdess(i) = 5; % Angry
        case '06'
            emotions_ravdess(i) = 6; % Fearful
        case '07'
            emotions_ravdess(i) = 7; % Disgust
        case '08'
            emotions_ravdess(i) = 8; % Surprised
    end
    
    % Convertir a one-hot encoding
    emotions_onehot_ravdess(i, emotions_ravdess(i)) = 1;

    
end



% Guardo las etiquetas tanto de numeros naturales como one hot
save('emotions_ravdess.mat', 'emotions_ravdess');
save('emotions_onehot_ravdess.mat', 'emotions_onehot_ravdess');


%% Creo la matriz de etiquetas con las 8 emociones a partir de la base de datos CREMAD

% Ruta de los archivos de audio de la base de datos CREMAD
audios_cremad = dir('C:\Users\maria.martinez\OneDrive - VINCI Energies\Documentos\Nueva carpeta\AudioWAV\*.wav');

% Creo una matriz vacía para almacenar las etiquetas
emotions_cremad = zeros(length(audios_cremad), 1);

% Creo una matriz para almacenar las etiquetas en formato one-hot porque si
% no en el entrenamiento me da problemas
numEmotions_cremad = 8; % Número total de emociones
emotions_onehot_cremad = zeros(length(audios_cremad), numEmotions_cremad);

for i = 1:length(audios_cremad)
    filename = audios_cremad(i).name;
    
    % Obtengo la etiqueta del archivo de audio
    emocion = filename(10:12);  
    

    % Asigna valores numéricos a las emociones
    switch emocion
        case 'HAP'
            emotions_cremad(i) = 3; % Alegría
        case 'SAD'
            emotions_cremad(i) = 4; % Tristeza
        case 'NEU'
            emotions_cremad(i) = 1; % Neutral
        case 'ANG'
            emotions_cremad(i) = 5; % Enfado
        case 'DIS'
            emotions_cremad(i) = 7; % Disgusto
        case 'FEA'
            emotions_cremad(i) = 6; % Miedo
        otherwise
            disp('Emoción no reconocida');
    end

    
    % Convertir a one-hot encoding
    emotions_onehot_cremad(i, emotions_cremad(i)) = 1;

end


% Guardo las etiquetas tanto de numeros naturales como one hot
save('emotions_cremad.mat', 'emotions_cremad');
save('emotions_onehot_cremad.mat', 'emotions_onehot_cremad');


%% Creo la matriz de etiquetas con las 8 emociones a partir de la base de datos SAVEE


% Ruta de los archivos de audio de la base de datos SAVEE
audios_savee = dir('C:\Users\maria.martinez\OneDrive - VINCI Energies\Documentos\Nueva carpeta\ALL\*.wav');

% Creo una matriz vacía para almacenar las etiquetas
emotions_savee = zeros(length(audios_savee), 1);

% Creo una matriz para almacenar las etiquetas en formato one-hot porque si
% no en el entrenamiento me da problemas
numEmotions_savee = 8; % Número total de emociones
emotions_onehot_savee = zeros(length(audios_savee), numEmotions_savee);

for i = 1:length(audios_savee)
    filename = audios_savee(i).name;
    
    % Obtengo la etiqueta del archivo de audio
    if length(filename) >= 11
        emocion = filename(4:5);  % Toma los dos caracteres en la posición 4 y 5 si están presentes
    else
        emocion = filename(4);  % Toma solo el carácter en la posición 4 si no hay un segundo carácter
    end
    
    % Asigno valores del 1-8 a las emociones
    switch emocion
        case 'h'
            emotions_savee(i) = 3; % Alegria
        case 'sa'
            emotions_savee(i) = 4; % Tristeza
        case 'a'
            emotions_savee(i) = 5; % Angry
        case 'd'
            emotions_savee(i) = 7; % Disgust
        case 'f'
            emotions_savee(i) = 6; % Fearful
        case 'n'
            emotions_savee(i) = 1; % Neutral
        case 'su'
            emotions_savee(i) = 8; % Surprise
    end
    
    % Convertir a one-hot encoding
    emotions_onehot_savee(i, emotions_savee(i)) = 1;

end



% Guardar las etiquetas tanto de numeros naturales como one hot
save('emotions_savee.mat', 'emotions_savee');
save('emotions_onehot_savee.mat', 'emotions_onehot_savee');



%% Ajustar todos los audios de las 3 bases de datos al tamaño mínimo para poder combinarlos en una sola matriz


% Ruta de los archivos de audio ravdess
audios_ravdess_path = 'C:\Users\maria.martinez\OneDrive - VINCI Energies\Documentos\Nueva carpeta\1440audios\';

% Encuentra la longitud mínima de los archivos de audio
minLength = Inf;

% Obtengo todos los audios
audios_ravdess = dir(fullfile(audios_ravdess_path, '*.wav'));

for i = 1:length(audios_ravdess)
    % Lee audio y obtiene su longitud
    filename = fullfile(audios_ravdess_path, audios_ravdess(i).name);
    [audio, fs] = audioread(filename);

    audioLength = size(audio, 1);

    % Actualiza la longitud mínima
    if audioLength < minLength
        minLength = audioLength;
    end
end

% Inicializa la matriz para almacenar los audios recortados
audioMatrixmin_ravdess = zeros(minLength, length(audios_ravdess));

% Recorre todos los audios y recorta al tamaño mínimo
for i = 1:length(audios_ravdess)

    filename = fullfile(audios_ravdess_path, audios_ravdess(i).name);
    [audio, fs] = audioread(filename);

    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Recorta audio al tamaño mínimo
    audio = audio(1:minLength);

    % Almacena audio en la matriz de audios
    audioMatrixmin_ravdess(:, i) = audio;
end

save('audioMatrixmin_ravdess.mat', 'audioMatrixmin_ravdess', '-v7.3');





% Ruta de los archivos de audio CRemaD
audios_cremad_path = 'C:/Users/maria.martinez/OneDrive - VINCI Energies/Documentos/Nueva carpeta/AudioWAV/';

% Encuentra la longitud mínima de los archivos de audio
minLength = Inf;

% Obtengo lista de archivos en la carpeta
audios_cremad = dir(fullfile(audios_cremad_path, '*.wav'));

for i = 1:length(audios_cremad)
    % Lee audio y obtiene su longitud
    filename = fullfile(audios_cremad_path, audios_cremad(i).name);
    [audio, fs] = audioread(filename);

    audioLength = size(audio, 1);

    % Actualiza la longitud mínima
    if audioLength < minLength
        minLength = audioLength;
    end
end

% Inicializa la matriz para almacenar los audios recortados
audioMatrixmin_cremad = zeros(minLength, length(audios_cremad));

% Recorre todos los audios y recorta al tamaño mínimo
for i = 1:length(audios_cremad)
    filename = fullfile(audios_cremad_path, audios_cremad(i).name);
    [audio, fs] = audioread(filename);

    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Recorta audio al tamaño mínimo
    audio = audio(1:minLength);

    % Almacena audio en la matriz de audios
    audioMatrixmin_cremad(:, i) = audio;
end

save('audioMatrixmin_cremad.mat', 'audioMatrixmin_cremad', '-v7.3');





% Ruta de los archivos de audio SAVEE
audios_savee_path = 'C:\Users\maria.martinez\OneDrive - VINCI Energies\Documentos\Nueva carpeta\ALL\';

% Encuentra la longitud mínima de los archivos de audio
minLength = Inf;

% Obtengo lista de archivos en la carpeta
audios_savee = dir(fullfile(audios_savee_path, '*.wav'));

for i = 1:length(audios_savee)
    % Lee audio y obtiene su longitud
    filename = fullfile(audios_savee_path, audios_savee(i).name);
    [audio, fs] = audioread(filename);

    audioLength = size(audio, 1);

    % Actualiza la longitud mínima
    if audioLength < minLength
        minLength = audioLength;
    end
end

% Inicializa la matriz para almacenar los audios recortados
audioMatrixmin_savee = zeros(minLength, length(audios_savee));

% Recorre todos los audios y recorta al tamaño mínimo
for i = 1:length(audios_savee)
    filename = fullfile(audios_savee_path, audios_savee(i).name);
    [audio, fs] = audioread(filename);

    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Recorta audio al tamaño mínimo
    audio = audio(1:minLength);

    % Almacena audio en la matriz de audios
    audioMatrixmin_savee(:, i) = audio;
end

save('audioMatrixmin_savee.mat', 'audioMatrixmin_savee', '-v7.3');


% Encuentra la longitud mínima entre las 3 matrices de audios
minLength = min([size(audioMatrixmin_ravdess, 1), size(audioMatrixmin_cremad, 1), size(audioMatrixmin_savee, 1)]);

% Recortar las 3 matrices nuevamente al tamaño mínimo
audioMatrixmin_ravdess = audioMatrixmin_ravdess(1:minLength, :);
audioMatrixmin_cremad = audioMatrixmin_cremad(1:minLength, :);
audioMatrixmin_savee = audioMatrixmin_savee(1:minLength, :);

% Creo la matriz de audios combinada y la matriz de etiquetas combinada

combinedData = [audioMatrixmin_ravdess, audioMatrixmin_cremad, audioMatrixmin_savee];
combinedLabels = [emotions_onehot_ravdess; emotions_onehot_cremad; emotions_onehot_savee];


%% División de los datos en 3 conjuntos + 1 extra de validación


% Numero total de muestras
totalSamplesCombined = size(combinedData, 2);

rng('default'); % Establece semilla aleatoria para reproducibilidad
indicesCombined = randperm(totalSamplesCombined);


% Separo300 audios para la evaluación posterior
num_audios_evaluacion_combined = 300;  
indices_evaluacion_combined = indicesCombined(1:num_audios_evaluacion_combined);
audios_evaluacion_combined = combinedData(:, indices_evaluacion_combined);
evaluationLabels = combinedLabels(indices_evaluacion_combined, :);


% Utilizo los índices aleatorios restantes para dividir los datos
remaining_indices_combined = setdiff(indicesCombined, indices_evaluacion_combined);
% Reorganiza los índices restantes de forma aleatoria
remaining_indices_combined = remaining_indices_combined(randperm(length(remaining_indices_combined)));


% Divido los datos combinados en 3 conjuntos
trainingRatio = 0.7;
validationRatio = 0.15;
testRatio = 0.15;


% Calculo tamaños de conjuntos
trainingSizeCombined = round(trainingRatio * (totalSamplesCombined - num_audios_evaluacion_combined));
validationSizeCombined = round(validationRatio * (totalSamplesCombined - num_audios_evaluacion_combined));
testSizeCombined = round(testRatio * (totalSamplesCombined - num_audios_evaluacion_combined));

% Utilizo los índices aleatorios para dividir los datos
trainingIndicesCombined = remaining_indices_combined(1:trainingSizeCombined);
validationIndicesCombined = remaining_indices_combined(trainingSizeCombined+1:trainingSizeCombined+validationSizeCombined);
testIndicesCombined = remaining_indices_combined(trainingSizeCombined+validationSizeCombined+1:end);

% Divide la matriz de datos combinada
trainingDataCombined = combinedData(:, trainingIndicesCombined);
validationDataCombined = combinedData(:, validationIndicesCombined);
testDataCombined = combinedData(:, testIndicesCombined);


% Creo matrices one-hot encoding para las etiquetas
numClasses = 8;

trainingLabelsCombined = zeros(trainingSizeCombined, numClasses);
validationLabelsCombined = zeros(validationSizeCombined, numClasses);
testLabelsCombined = zeros(testSizeCombined, numClasses);


% matrices one-hot encoding para las etiquetas de entrenamiento
for i = 1:trainingSizeCombined
    trainingLabelsCombined(i, :) = combinedLabels(trainingIndicesCombined(i), :);
end

% matrices one-hot encoding para las etiquetas de validación
for i = 1:validationSizeCombined
    validationLabelsCombined(i, :) = combinedLabels(validationIndicesCombined(i), :);
end

% matrices one-hot encoding para las etiquetas de prueba
for i = 1:testSizeCombined
    testLabelsCombined(i, :) = combinedLabels(testIndicesCombined(i), :);

end


% Verifico tamaños de conjuntos
disp(['Conjunto de entrenamiento: ' num2str(trainingSizeCombined) ' muestras']);
disp(['Conjunto de validación: ' num2str(validationSizeCombined) ' muestras']);
disp(['Conjunto de prueba: ' num2str(testSizeCombined) ' muestras']);
disp(['Conjunto de evaluación: ' num2str(num_audios_evaluacion_combined) ' muestras']);


%% Entrenamiento y ajuste de parámetros de la red neuronal

% Creo la red neuronal patternnet
hiddenLayerSizes = [500, 500, 250, 60];
net2 = patternnet(hiddenLayerSizes);

% Configurar las funciones de activación para cada capa
net2.layers{1}.transferFcn = 'logsig';
net2.layers{2}.transferFcn = 'logsig';
net2.layers{3}.transferFcn = 'logsig';
net2.layers{4}.transferFcn = 'logsig';
net2.layers{end}.transferFcn = 'softmax';
% Configurar la entrada y salida de la red
%net2.inputs{1}.size = size(trainingDataCombined, 2); % Tamaño de la
%entrada % por defecto, el tamaño es el de la entrada matriz de
%características
net2.layers{end}.size = 8; % Tamaño de la salida 


% Entrenar la red con entrenador 'trainlm' y ajustar manualmente la regularización L2
lambda = 0.01; % Parámetro de regularización L2
net2.performParam.regularization = lambda;
% parámetros de entrenamiento
net2.trainParam.epochs = 10000;  % Número de épocas
net2.trainParam.goal = 0;  % Objetivo de rendimiento
net2.trainParam.lr = 0.001;  % Tasa de aprendizaje
net2.trainParam.min_grad = 0;  % Gradiente mínimo
net2.trainParam.max_fail = 1000000000;  % Número máximo de fallos consecutivos

% Entreno la red neuronal
net2 = train(net2, trainingDataCombined, trainingLabelsCombined');

% Guardo la red 
save('red_neuronal_entrenada2.mat', 'net2');



%% Evaluación de la red neuronal en el conjunto de entrenamiento


% Utilizo la red neuronal para predecir etiquetas
predictions = net2(trainingDataCombined);
predictedLabels = vec2ind(predictions)';

% etiquetas reales del conjunto extra de validación
trueLabels = vec2ind(trainingLabelsCombined')';
% calculo la precisión
accuracy = sum(predictedLabels == trueLabels) / length(trueLabels) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto de entrenamiento: ' num2str(accuracy) '%']);




%% Evaluación de la red neuronal en el conjunto extra de validación

% Utilizo la red neuronal para predecir etiquetas
predictions2 = net2(audios_evaluacion_combined);
predictedLabels2 = vec2ind(predictions2)';


% etiquetas reales del conjunto extra de validación
trueLabels2 = vec2ind(evaluationLabels')';
% calculo la precisión
accuracy2 = sum(predictedLabels2 == trueLabels2) / length(trueLabels2) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto extra de validación: ' num2str(accuracy2) '%']);



%% Evaluación de la red neuronal en el conjunto de validación

% Utilizo la red neuronal para predecir etiquetas
predictions = net2(validationDataCombined);
predictedLabels = vec2ind(predictions)';

% etiquetas reales del conjunto extra de validación
trueLabels = vec2ind(validationLabelsCombined')';
% calculo la precisión
accuracy = sum(predictedLabels == trueLabels) / length(trueLabels) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto de validación: ' num2str(accuracy) '%']);
