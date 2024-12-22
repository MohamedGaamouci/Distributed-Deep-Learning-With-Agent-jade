package com.jade.ddl.Agents;

import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DataDistributorAgent extends Agent {

    @Override
    protected void setup() {
        System.out.println(getLocalName() + " is ready.");
        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {
                ACLMessage msg = receive();
                if (msg != null && msg.getPerformative() == ACLMessage.REQUEST) {
                    String content = msg.getContent();
                    if (content.startsWith("Split Data")) {
                        System.out.println(getLocalName() + " is splitting data...  .");
                        // Simulate data splitting
                        int num_worker = extractNumberFromMessage(content);
                        String SplitConfig = "{"
                                + "\"dataset\": \"mnist\","
                                + "\"num_worker\":"+num_worker
                                + "}";
                        try {
                            // Set up the connection
                            URL url = new URL("http://localhost:5000/split");
                            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                            conn.setRequestMethod("POST");
                            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                            conn.setDoOutput(true);

                            // Write the JSON payload to the output stream
                            try (OutputStream os = conn.getOutputStream()) {
                                os.write(SplitConfig.getBytes("UTF-8"));
                                os.flush();
                            }

                            // Read the server's response
                            int responseCode = conn.getResponseCode();
                            if (responseCode == 200) {
                                System.out.println("Training request sent successfully!");
                            } else {
                                System.err.println("Server responded with code: " + responseCode);
                            }
                        } catch (MalformedURLException e) {
                            e.printStackTrace();
                        } catch (ProtocolException e) {
                            throw new RuntimeException(e);
                        } catch (UnsupportedEncodingException e) {
                            throw new RuntimeException(e);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                        ACLMessage reply = msg.createReply();
                        reply.setPerformative(ACLMessage.INFORM);
                        reply.setContent("Data Split Complete");
                        send(reply);
                    } else if (content.equals("Distribute The Data")) {
                        System.out.println(getLocalName() + " is distributing data... .");
                        for (int i = 1; i <= 3; i++) {
                            ACLMessage dataMsg = new ACLMessage(ACLMessage.INFORM);
                            dataMsg.addReceiver(new AID("Worker" + i, AID.ISLOCALNAME));
                            dataMsg.setContent("Data Split for Worker" + i);
                            send(dataMsg);
                        }
                    }
                } else {
                    block();
                }
            }
        });
    }

    public int extractNumberFromMessage(String messageContent) {
        Pattern pattern = Pattern.compile("\\d+");
        Matcher matcher = pattern.matcher(messageContent);
        if (matcher.find()) {
            int numWorkers = Integer.parseInt(matcher.group());
            return numWorkers;
        } else {
            System.out.println("No number found in the message.");
        }
        return 0;
    }
}
