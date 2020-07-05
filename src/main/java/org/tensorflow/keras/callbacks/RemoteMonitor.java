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

import java.io.PrintStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.json.JSONObject;

/**
 *
 * @author jbclarke
 */
public class RemoteMonitor extends Callback {

    public static final String DEFAULT_URL = "http://localhost:9000/publish/epoch/end";
    public static final String DEFAULT_FIELD = "data";
    private final URL url;
    private final String field;
    private final Map<String, String> headers;
    private final boolean sendAsJson;

    public RemoteMonitor() throws MalformedURLException {
        this(null, null, new URL(DEFAULT_URL), DEFAULT_FIELD, null, false);
    }

    public RemoteMonitor(Map<String, Object> params, Object model,
            URL url, String field, Map<String, String> headers, boolean sendAsJson) {
        super(params, model);
        this.url = url;
        this.field = field;
        this.headers = headers;
        this.sendAsJson = sendAsJson;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEpochEnd(int epoch, Map<String, Number> logs) {
        logs = logs == null ? Collections.EMPTY_MAP : logs;
        Map<String, Number> send = new HashMap<>();
        send.put("epoch", epoch);
        send.putAll(logs);
        String jsonString = toJsonString(send);

        try {
            if (this.sendAsJson) {
                post(jsonString);
            } else {
                postForm(jsonString);
            }
        } catch (Exception ex) {

        }

    }

    private void post(String sendStr) throws Exception {
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json; utf-8");
        con.setRequestProperty("Accept", "application/json");
        con.setDoOutput(true);
        try (PrintStream stream = new PrintStream(con.getOutputStream())) {
            stream.print(sendStr);
        }
    }

    private void postForm(String sendStr) throws Exception {
        Map<String, Object> postParams = new HashMap<>();
        postParams.put(this.field, sendStr);
        StringBuilder postData = new StringBuilder();
        for (Map.Entry<String, Object> param : postParams.entrySet()) {

            if (postData.length() != 0) {
                postData.append('&');
            }
            postData.append(URLEncoder.encode(param.getKey(), "UTF-8"));
            postData.append('=');
            postData.append(URLEncoder.encode(String.valueOf(param.getValue()), "UTF-8"));

        }
        byte[] postDataBytes = postData.toString().getBytes("UTF-8");

        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
        conn.setRequestProperty("Content-Length", String.valueOf(postDataBytes.length));
        conn.setDoOutput(true);
        conn.getOutputStream().write(postDataBytes);

    }

    private String toJsonString(Map<String, Number> send) {
        JSONObject obj = new JSONObject();
        send.keySet().forEach(key -> {
            Number val = send.get(key);
            if (val instanceof Double) {
                obj.put(key, val.doubleValue());
            } else if (val instanceof Float) {
                obj.put(key, val.floatValue());
            } else if (val instanceof Long) {
                obj.put(key, val.longValue());
            } else if (val instanceof Integer || val instanceof Short) {
                obj.put(key, val.intValue());
            }
        });
        return obj.toString();
    }

}
