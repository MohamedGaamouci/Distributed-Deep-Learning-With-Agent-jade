package com.jade.ddl.Agents;

import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class WorkerAgent extends Agent {

    private boolean hasSentModel = false; // Tracks whether the model has been sent

    @Override
    protected void setup() {
        System.out.println(getLocalName() + " is ready.");

        addBehaviour(new CyclicBehaviour() {
            @Override
            public void action() {
                ACLMessage msg = receive();
                if (msg != null) {
                    String content = msg.getContent();
                    if (content.startsWith("{") && content.endsWith("}")){
                        try {
                            // Set up the connection
                            int agent_number = extractNumberFromMessage(getLocalName());
                            URL url = new URL("http://localhost:5000/train/"+agent_number);
                            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                            conn.setRequestMethod("POST");
                            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                            conn.setDoOutput(true);

                            // Write the JSON payload to the output stream
                            try (OutputStream os = conn.getOutputStream()) {
                                os.write(content.getBytes("UTF-8"));
                                os.flush();
                            }

                            // Read the server's response
                            int responseCode = conn.getResponseCode();
                            if (responseCode == 200) {
                                System.out.println("Training Done Successfully!");
                                ACLMessage reply = msg.createReply();
                                reply.setPerformative(ACLMessage.INFORM);
                                reply.setContent("Training Complete.");
                                send(reply);
                            } else {
                                System.err.println("Server responded with code: " + responseCode);
                            }

                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        hasSentModel = false;
                    }else if (content.equals("Prepare for Training")) {
                        System.out.println(getLocalName() + " is preparing for training.");
                        ACLMessage reply = msg.createReply();
                        reply.setPerformative(ACLMessage.INFORM);
                        reply.setContent("Ready for Training");
                        send(reply);
                    } else if (content.startsWith("Data Split")) {
                        System.out.println(getLocalName() + " received: <<" + content + " >>");
                        System.out.println(getLocalName() + " is training.....");
//                        try {
//                            Thread.sleep(3000); // Simulate training
//                        } catch (InterruptedException e) {
//                            e.printStackTrace();
//                        }
                        ACLMessage reply = new ACLMessage(ACLMessage.INFORM);
                        reply.addReceiver(new AID("Coordinator", AID.ISLOCALNAME));
                        reply.setContent("Training Complete.");
                        send(reply);
                    } else if (content.equals("Send Model to Aggregator")) {
                        if (!hasSentModel) {
                            System.out.println(getLocalName() + " is sending model to aggregator.");
                            ACLMessage modelMsg = new ACLMessage(ACLMessage.INFORM);
                            modelMsg.addReceiver(new AID("ModelAggregator", AID.ISLOCALNAME));
                            modelMsg.setContent("Model from " + getLocalName());
                            send(modelMsg);

                            System.out.println(getLocalName() + " has successfully sent the model.");
                            hasSentModel = true; // Mark model as sent
                        } else {
                            System.out.println(getLocalName() + " has already sent the model. Ignoring duplicate request.");
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

    @Override
    protected void takeDown() {
        System.out.println(getLocalName() + " is shutting down.");
    }
}
