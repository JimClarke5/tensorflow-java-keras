/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.callbacks;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.tensorflow.keras.utils.ND;
import org.tensorflow.tools.ndarray.NdArray;

/**
 * Callback that streams epoch results to a csv file.
 */
public class CSVLogger extends Callback implements AutoCloseable {

    private final String filename;
    private final String separator;
    private final boolean append;
    private List<String> keys;
    private boolean appendHeader = true;

    private CSVPrinter writer;

    /**
     * Creates a CSVLogger callback.
     *
     * @param filename filename of the csv file
     */
    public CSVLogger(String filename) {
        this(filename, ",", false);
    }

    /**
     * Creates a CSVLogger callback.
     *
     * @param filename filename of the csv file
     * @param separator string used to separate elements in the csv file.
     */
    public CSVLogger(String filename, String separator) {
        this(filename, separator, false);
    }

    /**
     * Creates a CSVLogger callback.
     *
     * @param filename filename of the csv file
     * @param separator string used to separate elements in the csv file.
     * @param append if true, append if file exists (useful for continuing
     * training). if false, overwrite existing file,
     */
    public CSVLogger(String filename, String separator, boolean append) {
        this.filename = filename;
        this.separator = separator;
        this.append = append;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrainBegin(Map<String, Number> logs) {
        File file = new File(this.filename);
        try {
            if (this.append) {
                if (file.exists()) {
                    try (CSVParser parser = new CSVParser(new FileReader(file), CSVFormat.EXCEL)) {
                        for (CSVRecord record : parser) {
                            appendHeader = false;
                            break;
                        }
                    } catch (IOException ex) {
                        Logger.getLogger(CSVLogger.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            }
            writer = new CSVPrinter(new FileWriter(filename, append), CSVFormat.EXCEL);
        } catch (IOException ex) {
            Logger.getLogger(CSVLogger.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    //TODO Should we handle Java arrays??
    private String handleValue(Object val) {
        boolean is_zero_dim_ndarray = val instanceof NdArray && ((NdArray) val).rank() == 0;
        if (val instanceof String) {
            return val.toString();
        } else if (val instanceof NdArray) {  // todo
            return ND.toString((NdArray) val);
        } else if (val instanceof Collection) {
            return "[" + ((Collection) val).stream().map(Object::toString).collect(Collectors.joining(",")) + "]";
        } else {
            return val.toString();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;

        if (this.keys == null) {
            this.keys = new ArrayList<String>(logs.keySet());
            Collections.sort(this.keys);
        }

        /**
         * TODO if(this.model.stopTraining) { this.keys.forEach(key -> {
         * if(!logs.containsKey(key)) { logs.put(key, Double.NaN); } }); } *
         */
        try {
            if (this.writer == null) {

                List<String> fieldNames = new ArrayList<>();
                fieldNames.add("epoch");
                fieldNames.addAll(this.keys);
                writer = new CSVPrinter(new FileWriter(filename, append), CSVFormat.EXCEL);
                if (this.appendHeader) {
                    writer.printRecord(fieldNames);
                }

            }
            final List values = new ArrayList();
            final Map<String, Number> logsFinal = logs;
            values.add(epoch);
            keys.forEach(key -> {
                values.add(handleValue(logsFinal.get(key)));
            });
            writer.printRecord(values);
            writer.flush();
        } catch (IOException ex) {
            Logger.getLogger(CSVLogger.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() throws IOException {
        if (writer != null) {
            writer.close();
            writer = null;
        }
    }

}
