let dashcore = require('@dashevo/dashcore-lib');
let request = require('requests');
let got = require('got');
var sender = 'yTYZjnTuepHbVAcoWq4g7f5teXru4KSJMa'
var receiver = 'yNpEzKCvS2Vn3WYhXeG11it5wEWMButDvq'
var senderPrivatekey = 'adb27adb845cf776e49ba7f09e58cf53182fcdfd5c3c1ac919340117b41e1a7b'
let token = '8PGdgeEbzxm7SvMWdM4MBIJU5lvnL2w7'
let url = `https://api.chainrider.io/v1/dash/testnet/addr/${sender}/utxo?token=${token}`
let send_amount = 20000
let raw


//  GRADED FUNCTION
//  TASK-1: Write a function that sends {send_amount} of dash from {sender} to {receiver}.
//  Register on ChainRider to get a ChainRider token (instructions provided) and input its value as {token}
//  Create a transaction using the {dashcore} library, and send the transaction using ChainRider
//  Send Raw Transaction API - https://www.chainrider.io/docs/dash/#send-raw-transaction
//  The resulting transaction ID is needed to be supplied through the Assignment on Coursera


//FIRST GET REQUEST : get utxo_obj from URL

 let utxo_obj = {"address":"yTYZjnTuepHbVAcoWq4g7f5teXru4KSJMa",
           "txid":"e9b25b28e08c86a3d1658b1bd5138356f502a863f25f1ac2d73f6440192a62b0",
            "vout":1,
             "scriptPubKey":"76a9144f4409f6a42a8f3ff565e4d1df47e8e8b1d509cb88ac",
                "amount":0.00039,"satoshis":39000,
                "height":401803,"confirmations":31};

// (async () => {
//     try {
//         // GET request
//         let response = await got(url);
//         let body = response.body;
//         let obj = JSON.parse(body)[0];
//         console.log('obj:', obj);
//         let amount = obj.amount;
//         console.log('GET response:', response.body);
//         console.log('Amount:', amount);
//
//         let transaction = new dashcore.Transaction()
//             .from(utxo_obj)
//             .to(receiver, 20000)
//             .change(sender)
//             .sign(senderPrivatekey)
//
//         rawtxserial = transaction.serialize();
//         console.log('rawtxSerial:', rawtxserial);
//
//         let postUrl = 'https://api.chainrider.io/v1/dash/testnet/tx/send';
//         const options = {
//             body: [{
//                 rawtx: rawtxserial,
//                 token: token
//             }],
//             url : 'https://api.chainrider.io/v1/dash/testnet/tx/send',
//             json: true
//         };
//
//         let post = await got.post(postUrl, options);
//         console.log('POST: ', post.body);
//
//     } catch (error) {
//     }
// })();

// const request = require('request');



// const options = {
//     url: 'https://api.chainrider.io/v1/dash/testnet/tx/send',
//     json: true,
//     body: {
//         rawtx: rawtxserial,
//         token: token
//     }
// };
// request.post(options, (err, res, body) => {
//     if (err) {
//         return console.log(err);
//     }
//     console.log("the body is",body);
// });

// const got = require('got');
//
(async () => {
    try{
        let transaction = new dashcore.Transaction()
            .from(utxo_obj)
            .to(receiver, 20000)
            .change(sender)
            .sign(senderPrivatekey)

        rawtxserial = transaction.serialize();
        console.log('HEx tx: ', rawtxserial);

        url =  'https://api.chainrider.io/v1/dash/testnet/tx/' + utxo_obj.txid;


        headers = {
            'Content-Type':'application/json',
            'Accept':'application/json',
        }

        console.log()
        r = await request.get(url,
            params={'token': token}, headers = headers);

        console.log('JSON response: ', r.json());

        // response = await got(url);
        // console.log('Response: ',response);
        // body = response.body;
        // obj = JSON.parse(body)[0];
        // console.log('Transaction with vout:', obj);
    }catch(error){
        console.log('Error: ', error);
    }
})();


console.log('stringify result: ', JSON.stringify( {name:'Ioana'}));

// POST request
//     const options = {
//         body: [{
//             rawtx: rawtxserial,
//             token: token
//         }],
//         url : 'https://api.chainrider.io/v1/dash/testnet/tx/send'
//     };
// (async () => {
//     try {
//
//         let post = await got.post(url, options);
//         console.log('POST: ', post);
//     } catch(error){
//     }
//     // console.log(body.data);
//     //=> {hello: 'world'}
// })();
