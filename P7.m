% P4 (Sin preprocesamiento, solo normalizado) le añado regularización L2


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


%%
% Suponiendo que tienes las matrices de audio audioMatrixmin_cremad y audioMatrixmin_savee

% Filtrar el ruido de fondo de las matrices de audio
audio_filtrado_cremad = zeros(size(audioMatrixmin_cremad));
audio_filtrado_savee = zeros(size(audioMatrixmin_savee));

% Filtrar el ruido de fondo de la matriz audioMatrixmin_cremad
for i = 1:size(audioMatrixmin_cremad, 1)
    audio_filtrado_cremad(i, :) = medfilt1(audioMatrixmin_cremad(i, :), 3); 
end

% Filtrar el ruido de fondo de la matriz audioMatrixmin_savee
for i = 1:size(audioMatrixmin_savee, 1)
    audio_filtrado_savee(i, :) = medfilt1(audioMatrixmin_savee(i, :), 3);
end

% Guardar las matrices filtradas en archivos separados (opcional)
save('audioMatrixmin_cremad_filtrado.mat', 'audio_filtrado_cremad');
save('audioMatrixmin_savee_filtrado.mat', 'audio_filtrado_savee');




%% Creo la matriz de audios combinada y la matriz de etiquetas combinada

combinedData = [audioMatrixmin_ravdess, audio_filtrado_cremad, audio_filtrado_savee];
combinedLabels = [emotions_onehot_ravdess; emotions_onehot_cremad; emotions_onehot_savee];




%%

% EXTRACCIÓN DE CARACTERÍSTICAS con los siguientes parámetros globales
windowSize = 2048; % Tamaño de la ventana en muestras (2048)
hopSize = 512; % Tamaño del paso en muestras (512)
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
    
    % Preénfasis de la señal (probarrr)
    preEmphasized = filter([1 -0.97], 1, audio);
    
    %frames = buffer(preEmphasized, windowSize, hopSize, 'nodelay');
    % Obtiene el número total de frames
    %numFrames = size(frames, 2);
        
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

    % Calcula la transformada de Fourier
    fft_audio = fft(audio);

    % Obtiene el espectro de frecuencias
    frecuencias = abs(fft_audio);

    % Frecuencia fundamental
    [~, idx] = max(frecuencias);
    f0 = (idx-1) * fs / length(audio);

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

     % Calcula la tasa de cruces por cero
    if isempty(audio)
        % Manejo del caso donde la señal es vacía o muy corta
        zcrMatrix(i) = 0;  
    else
        zcrValue = sum(abs(diff(sign(audio)))) / (2 * length(audio));
        % Verifica si el resultado es NaN
        if isnan(zcrValue)
             zcrMatrix(i) = 0; % Asigna un valor específico en lugar de NaN
        else
            % Almacena la tasa de cruces por cero en la matriz
            zcrMatrix(i) = zcrValue;
        end

    end
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


    % Ajusta la longitud de la ventana si es necesario
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

   % Verifica si la señal es vacía o muy corta
    % Verifica si la señal es vacía o muy corta
    if isempty(audio)
        % Manejo del caso donde la señal es vacía o muy corta
        rmsMatrix(i) = 0;  
    else
        % Calcula el RMS
        rmsValue = sqrt(mean(audio.^2));

        % Verifica si el resultado es NaN
        if isnan(rmsValue)

            % Asigna un valor específico en lugar de NaN
            rmsMatrix(i) = 0;  
        else
            % Almacena el valor RMS en la matriz
            rmsMatrix(i) = rmsValue;
        end
    end
end

% Guarda la matriz de valores RMS
save('rmsMatrix.mat', 'rmsMatrix', '-v7.3');




%% Entropy
% Matriz para almacenar los valores de entropía
entropies = zeros(1, size(combinedData, 2));

for i = 1:size(combinedData, 2)
    % Obtener el audio
    audio = combinedData(:, i);

    % Verificar si la señal está vacía o muy corta
    if isempty(audio)
        % Asignar un valor específico para audios vacíos o cortos
        entropies(i) = 0;  
    else
        % Calcular el histograma del audio
        [counts, ~] = histcounts(audio, 'Normalization', 'probability');
        
        % Calcular la entropía
        entropyValue = -sum(counts .* log2(counts + eps));  % eps se utiliza para evitar log(0)

        % Verificar si el resultado es NaN
        if isnan(entropyValue)
            % Asignar un valor específico en lugar de NaN
            entropies(i) = 0;  
        else
            % Almacenar el valor de entropía en la matriz
            entropies(i) = entropyValue;
        end
    end
end

% Guardar la matriz de entropías
save('entropies.mat', 'entropies', '-v7.3');




%% Concatenar las matrices

allFeaturesMatrix = [chromaMatrix; zcrMatrix; mfccMatrix; rmsMatrix; entropies];

% Transponer la matriz para tener una estructura adecuada (características en columnas)
allFeaturesMatrix = allFeaturesMatrix';

% Guardar la matriz resultante
save('allFeaturesMatrix.mat', 'allFeaturesMatrix', '-v7.3');




 %% otra normalizacion
% % Transponer la matriz para tener una estructura adecuada (características en columnas)
allFeaturesMatrix = allFeaturesMatrix';
% 
% % Normalizar por columna (característica)
normalizedFeaturesMatrix = zscore(allFeaturesMatrix);
% 
% % Guardar la matriz normalizada
save('normalizedFeaturesMatrix.mat', 'normalizedFeaturesMatrix', '-v7.3');


%% División de los datos en 3 conjuntos + 1 extra de validación

% Numero total de muestras
totalSamplesFeatures = size(normalizedFeaturesMatrix', 1);

rng('default'); % Establece semilla aleatoria para reproducibilidad
indicesFeatures = randperm(totalSamplesFeatures);

% Separar algunas muestras para la evaluación posterior
numSamplesEvaluation = 500;  
indicesEvaluation = indicesFeatures(1:numSamplesEvaluation);
featuresEvaluation = normalizedFeaturesMatrix(:, indicesEvaluation);
evaluationLabels = combinedLabels(indicesEvaluation, :);



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

normalizedFeaturesMatrix = normalizedFeaturesMatrix';

% Divide la matriz de características
trainingFeatures = normalizedFeaturesMatrix(trainingIndicesFeatures, :);
validationFeatures = normalizedFeaturesMatrix(validationIndicesFeatures, :);
testFeatures = normalizedFeaturesMatrix(testIndicesFeatures, :);


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


%% Entrenamiento y ajuste de parámetros de la red neuronal

% Creo la red neuronal patternnet
hiddenLayerSizes = [512 256 64];
net7b = patternnet(hiddenLayerSizes);

net7b.layers{end}.size = 8; % Tamaño de la salida (8 en este caso)


% Entrenar la red con entrenador 'trainlm' y ajustar manualmente la regularización L2
% Añadir regularización
lambda = 0.01; % Parámetro de regularización L2
net7b.performParam.regularization = lambda;
% Configurar la función de activación ReLU para cada capa oculta, LA DE
% SALIDA SOFTMAX POR DEFECTO
net7b.layers{1}.transferFcn = 'poslin';
net7b.layers{2}.transferFcn = 'poslin';
net7b.layers{3}.transferFcn = 'poslin';

% parámetros de entrenamiento
net7b.trainParam.epochs = 6000;  % Número de épocas
net7b.trainParam.goal = 0;  % Objetivo de rendimiento
net7b.trainParam.lr = 0.001;  % Tasa de aprendizaje
net7b.trainParam.min_grad = 1e-6; % Criterio de convergencia basado en el gradiente
net7b.trainParam.max_fail = 1000000000;  % Número máximo de fallos consecutivos

% Entreno la red neuronal
net7b = train(net7b, trainingFeatures', trainingLabelsFeatures');

% Guardo la red 
save('red_neuronal_entrenada7b.mat', 'net7b');



%% Evaluación de la red neuronal en el conjunto de entrenamiento


% Utilizo la red neuronal para predecir etiquetas
predictions = net7(trainingFeatures');
predictedLabels = vec2ind(predictions)';

% etiquetas reales del conjunto extra de validación
trueLabels = vec2ind(trainingLabelsFeatures')';
% calculo la precisión
accuracy = sum(predictedLabels == trueLabels) / length(trueLabels) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto de entrenamiento: ' num2str(accuracy) '%']);




%% Evaluación de la red neuronal en el conjunto de validación

% Utilizo la red neuronal para predecir etiquetas
predictions = net7b(validationFeatures');
predictedLabels = vec2ind(predictions)';

% etiquetas reales del conjunto extra de validación
trueLabels = vec2ind(validationLabelsFeatures')';
% calculo la precisión
accuracy = sum(predictedLabels == trueLabels) / length(trueLabels) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto de validación: ' num2str(accuracy) '%']);



%% Evaluación de la red neuronal en el conjunto extra de validación

% Utilizo la red neuronal para predecir etiquetas
predictions2 = net7b(featuresEvaluation);
predictedLabels2 = vec2ind(predictions2);


% etiquetas reales del conjunto extra de validación
trueLabels2 = vec2ind(evaluationLabels');
% calculo la precisión
accuracy2 = sum(predictedLabels2 == trueLabels2) / length(trueLabels2) * 100;

disp(['Porcentaje de emociones correctas detectadas en el conjunto extra de validación: ' num2str(accuracy2) '%']);





%%
% Calcular la matriz de confusión en el conjunto de VALIDACION EXXTRA
confusionMatrix = confusionmat(trueLabels, predictedLabels);
disp('Matriz de Confusión en el conjunto extra de validación:');
disp(confusionMatrix);

%%

% Calcular la matriz de confusión en el conjunto de VALIDACION EXXTRA
confusionMatrix = confusionmat(trueLabels, predictedLabels);

% Crear una figura de la matriz de confusión normalizada
figure;
confusionchart(trueLabels, predictedLabels, 'Normalization', 'row-normalized');
title('Matriz de Confusión Normalizada en el Conjunto de Validación Extra');




%%
% Calcular precision, recall y specificity en el conjunto extra de
% validación

precision = confusionMatrix(1,1) / (confusionMatrix(1,1) + confusionMatrix(2,1) + confusionMatrix(3,1) + confusionMatrix(4,1) + confusionMatrix(5,1) + confusionMatrix(6,1) + confusionMatrix(7,1)+ confusionMatrix(8,1));
disp(['Precisión: ' num2str(precision)]);






