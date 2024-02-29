% P3: primer entrenamiento con matriz de características como entrada de
% la red

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



% Guardar las etiquetas tanto de numeros naturales como one hot
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
    
    % Asigno valores del 1-8 a las emociones
    % Asignar valores numéricos a las emociones
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


% Guardar las etiquetas tanto de numeros naturales como one hot
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



%%

% EXTRACCIÓN DE CARACTERÍSTICAS con los siguientes parámetros globales

windowSize = 2048; % Tamaño de la ventana en muestras 
hopSize = 512; % Tamaño del paso en muestras 
% Ventaneo de la señal (hsmming)
window = hamming(windowSize, 'periodic');


%% CHROMA

% Parámetros para el cálculo del cromagrama
numChroma = 12;  % Número de componentes del cromagrama


% Inicializa la matriz para almacenar los cromagramas
chromaMatrix = zeros(numChroma, size(combinedData, 2));

for i = 1:size(combinedData, 2)
    % Obtiene el audio recortado
    audio = combinedData(:, i);
    
    % Preénfasis de la señal 
    preEmphasized = filter([1 -0.97], 1, audio);
    
    
        
    % Calcula la STFT
    stftMatrix = spectrogram(preEmphasized, window, windowSize - hopSize, windowSize, fs);

    % Calcula las características croma a partir de la STFT
    chromaFeatures = sum(abs(stftMatrix(1:numChroma, :)).^2, 2);


    % Almacena las características croma en la matriz
    chromaMatrix(:, i) = chromaFeatures;
end

% Guarda la matriz de características
save('chromaMatrix.mat', 'chromaMatrix', '-v7.3');


%% F0

% Matriz para almacenar las frecuencias fundamentales
f0Matrix = zeros(1, size(combinedData, 2));

for i = 1:size(combinedData, 2)
    % Obtiene el audio recortado
    audio = combinedData(:, i);
    
    % Preénfasis de la señal
    preEmphasized = filter([1 -0.97], 1, audio);

   
    
    % Asegurar que preEmphasized y window tengan la misma longitud
    if length(preEmphasized) > length(window)
        preEmphasized = preEmphasized(1:length(window));
    else
        preEmphasized = [preEmphasized; zeros(length(window)-length(preEmphasized), 1)];
    end
    
    preEmphasized = preEmphasized .* window;

    % Calcula la transformada de Fourier
    fft_audio = fft(preEmphasized, hopSize);

    % Obtiene el espectro de frecuencias
    frecuencias = abs(fft_audio); % estoy considerando todo el espectro porque no se si los datos tienen informacion más significativas en las frecuencias altas o bajas
    %frecuencias = frecuencias(round(length(frecuencias)/2)+1:end); % se cortan las frecuencias más bajas del espectro
    %frecuencias = frecuencias(1:round(length(frecuencias)/2)) % se cortan las frecuencias más altas del espectro

    % Frecuencia fundamental
    [~, idx] = max(frecuencias);
    f0 = (idx-1) * fs / windowSize;

    % Almacena la frecuencia fundamental en la matriz
    f0Matrix(i) = f0;
end

% Guarda la matriz de frecuencias fundamentales
save('f0Matrix.mat', 'f0Matrix', '-v7.3');



%% ZCR (TASA DE CRUCES POR CERO)

% Matriz para almacenar las tasas de cruces por cero
zcrMatrix = zeros(1, size(combinedData, 2));

for i = 1:size(combinedData, 2)
    % Obtiene el audio recortado
    audio = combinedData(:, i);
    
    % Ventaneo de la señal
    windowedAudio = audio.* window;
    
    % Calcula la tasa de cruces por cero
    zcrValue = sum(abs(diff(sign(windowedAudio)))) / (2 * length(windowedAudio));
    
    % Almacena la tasa de cruces por cero en la matriz
    zcrMatrix(i) = zcrValue;
end

% Guardar la matriz de tasas de cruces por cero
save('zcrMatrix.mat', 'zcrMatrix', '-v7.3');


%% MFCC

numCoeffs = 13; % Número de coeficientes cepstrales

% Matriz para almacenar los MFCC
mfccMatrix = zeros(numCoeffs, size(combinedData, 2));

for i = 1:size(combinedData, 2)
    % Obtiene el audio recortado
    audio = combinedData(:, i);
    


    if length(audio) ~= windowSize
        window = hamming(length(audio));
        
    else
        window = hamming(windowSize);
    end
    

    % Aplica la ventana
    audio = audio .* window;

     % Calcula los MFCC utilizando la función mfcc
    mfccs = mfcc(audio, fs, 'Window', window, 'OverlapLength', windowSize - hopSize, 'NumCoeffs', numCoeffs);

    % Almacena los MFCC en la matriz
    mfccMatrix(:, i) = mean(mfccs, 2); % Puedes tomar la media de los coeficientes para obtener un único vector representativo
end

% Guardar la matriz de MFCC
save('mfccMatrix.mat', 'mfccMatrix', '-v7.3');



%% RMS (Root Mean Square)

% Matriz para almacenar los valores RMS
rmsMatrix = zeros(1, size(combinedData, 2));

for i = 1:size(combinedData, 2)
    % Obtiene el audio recortado
    audio = combinedData(:, i);
    
    % Aplica la ventana
    windowedAudio = audio .* window;
    
    % Calcula el RMS
    rmsValue = sqrt(mean(windowedAudio.^2));
    
    % Almacena el valor RMS en la matriz
    rmsMatrix(i) = rmsValue;
end

% Guarda la matriz de valores RMS
save('rmsMatrix.mat', 'rmsMatrix', '-v7.3');


%% Concatenar las matrices

allFeaturesMatrix = [chromaMatrix; f0Matrix; zcrMatrix; rmsMatrix];

% Transponer la matriz para tener una estructura adecuada (características en columnas)
allFeaturesMatrix = allFeaturesMatrix';

% Guardar la matriz resultante
save('allFeaturesMatrix.mat', 'allFeaturesMatrix', '-v7.3');



%% División de los datos en 3 conjuntos + 1 extra de validación

% Numero total de muestras
totalSamplesFeatures = size(allFeaturesMatrix, 1);

rng('default'); % Establece semilla aleatoria para reproducibilidad
indicesFeatures = randperm(totalSamplesFeatures);

% Separar algunas muestras para la evaluación posterior
numSamplesEvaluation = 500;  
indicesEvaluation = indicesFeatures(1:numSamplesEvaluation);
featuresEvaluation = allFeaturesMatrix(indicesEvaluation, :);

% Utilizo los índices aleatorios restantes para dividir los datos
remainingIndicesFeatures = setdiff(indicesFeatures, indicesEvaluation);
% Reorganiza los índices restantes de forma aleatoria
remainingIndicesFeatures = remainingIndicesFeatures(randperm(length(remainingIndicesFeatures)));

% Divido los datos en 3 conjuntos
trainingRatio = 0.7;
validationRatio = 0.15;
testRatio = 0.15;

% Calculo tamaños de conjuntos
trainingSizeFeatures = round(trainingRatio * (totalSamplesFeatures - numSamplesEvaluation));
validationSizeFeatures = round(validationRatio * (totalSamplesFeatures - numSamplesEvaluation));
testSizeFeatures = round(testRatio * (totalSamplesFeatures - numSamplesEvaluation));

% Utilizo los índices aleatorios para dividir los datos
trainingIndicesFeatures = remainingIndicesFeatures(1:trainingSizeFeatures);
validationIndicesFeatures = remainingIndicesFeatures(trainingSizeFeatures+1:trainingSizeFeatures+validationSizeFeatures);
testIndicesFeatures = remainingIndicesFeatures(trainingSizeFeatures+validationSizeFeatures+1:end);

% Divide la matriz de características
trainingFeatures = allFeaturesMatrix(trainingIndicesFeatures, :);
validationFeatures = allFeaturesMatrix(validationIndicesFeatures, :);
testFeatures = allFeaturesMatrix(testIndicesFeatures, :);


% Creo matrices one-hot encoding para las etiquetas (reemplázalo con tus etiquetas reales)
numClasses = 8;

trainingLabelsFeatures = zeros(trainingSizeFeatures, numClasses);
validationLabelsFeatures = zeros(validationSizeFeatures, numClasses);
testLabelsFeatures = zeros(testSizeFeatures, numClasses);

% Matrices one-hot encoding para las etiquetas de entrenamiento
for i = 1:trainingSizeFeatures
    % Reemplaza con tus etiquetas reales
    trainingLabelsFeatures(i, :) = combinedLabels(trainingIndicesFeatures(i), :);
end

% Matrices one-hot encoding para las etiquetas de validación
for i = 1:validationSizeFeatures
    % Reemplaza con tus etiquetas reales
    validationLabelsFeatures(i, :) = combinedLabels(validationIndicesFeatures(i), :);
end

% Matrices one-hot encoding para las etiquetas de prueba
for i = 1:testSizeFeatures
    % Reemplaza con tus etiquetas reales
    testLabelsFeatures(i, :) = combinedLabels(testIndicesFeatures(i), :);
end

% Verifico tamaños de conjuntos
disp(['Conjunto de entrenamiento: ' num2str(trainingSizeFeatures) ' muestras']);
disp(['Conjunto de validación: ' num2str(validationSizeFeatures) ' muestras']);
disp(['Conjunto de prueba: ' num2str(testSizeFeatures) ' muestras']);
disp(['Conjunto de evaluación: ' num2str(numSamplesEvaluation) ' muestras']);


%%
% Calcular la distribución de clases en el conjunto de entrenamiento
classDistributionTraining = sum(trainingLabelsFeatures);
disp('Distribución de clases en el conjunto de entrenamiento:');
disp(classDistributionTraining);

% Calcular la distribución de clases en el conjunto de validación
classDistributionValidation = sum(validationLabelsFeatures);
disp('Distribución de clases en el conjunto de validación:');
disp(classDistributionValidation);

% Calcular la distribución de clases en el conjunto de prueba
classDistributionTest = sum(testLabelsFeatures);
disp('Distribución de clases en el conjunto de prueba:');
disp(classDistributionTest);

%%


% Imprimir la distribución de clases antes de la división
disp('Distribución de clases antes de la división:');
classDistribution = sum(combinedLabels);
disp(classDistribution);



%% Entrenamiento y ajuste de parámetros de la red neuronal

% Creo la red neuronal patternnet
hiddenLayerSizes = [500, 500, 250, 60];
net3 = patternnet(hiddenLayerSizes);

net3.layers{end}.size = 8; % Tamaño de la salida (8 en este caso)

net3.trainFcn = 'trainlm';  % Establece el algoritmo de entrenamiento a 'trainlm'


% parámetros de entrenamiento
net3.trainParam.epochs = 10000;  % Número de épocas
net3.trainParam.goal = 0;  % Objetivo de rendimiento
net3.trainParam.lr = 0.001;  % Tasa de aprendizaje
net3.trainParam.min_grad = 1e-6; % Criterio de convergencia basado en el gradiente
net3.trainParam.max_fail = 1000000000;  % Número máximo de fallos consecutivos

% Entreno la red neuronal
net3 = train(net3, trainingFeatures', trainingLabelsFeatures');

% Guardo la red 
save('red_neuronal_entrenada3.mat', 'net3');




%% Evaluación de la red neuronal en el conjunto de entrenamiento


% Utilizo la red neuronal para predecir etiquetas
predictions = net3(trainingFeatures');
predictedLabels = vec2ind(predictions)';

% etiquetas reales del conjunto extra de validación
trueLabels = vec2ind(trainingLabelsFeatures')';
% calculo la precisión
accuracy = sum(predictedLabels == trueLabels) / length(trueLabels) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto de entrenamiento: ' num2str(accuracy) '%']);


%% Evaluación de la red neuronal en el conjunto extra de validación

% Utilizo la red neuronal para predecir etiquetas
predictions2 = net3(audios_evaluacion_combined');
predictedLabels2 = vec2ind(predictions2)';


% etiquetas reales del conjunto extra de validación
trueLabels2 = vec2ind(evaluationLabels');
% calculo la precisión
accuracy2 = sum(predictedLabels2 == trueLabels2) / length(trueLabels2) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto extra de validación: ' num2str(accuracy2) '%']);



%% Evaluación de la red neuronal en el conjunto de validación

% Utilizo la red neuronal para predecir etiquetas
predictions = net3(validationFeatures');
predictedLabels = vec2ind(predictions)';

% etiquetas reales del conjunto extra de validación
trueLabels = vec2ind(validationLabelsFeatures')';
% calculo la precisión
accuracy = sum(predictedLabels == trueLabels) / length(trueLabels) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto de validación: ' num2str(accuracy) '%']);


%% Evaluación de la red neuronal en el conjunto de prueba

% Utilizo la red neuronal para predecir etiquetas
predictions = net3(testFeatures');
predictedLabels = vec2ind(predictions)';

% etiquetas reales del conjunto extra de validación
trueLabels = vec2ind(testLabelsFeatures')';
% calculo la precisión
accuracy = sum(predictedLabels == trueLabels) / length(trueLabels) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto de validación: ' num2str(accuracy) '%']);
