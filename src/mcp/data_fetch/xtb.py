import websocket, json, openpyxl
from websocket import WebSocketConnectionClosedException
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

load_dotenv()
type_mapping_inverse = {
    'order_open': 0,
    'order_pending': 1,
    'order_close': 2,
    'order_modify': 3,
    'order_delete': 4
}


class XTB:
    __version__ = "1.0"

    def __init__(self):
               # Initialize the XTB client with credentials from environment variables
        self.ID = os.getenv('API_USER')
        self.PSW = os.getenv('API_PASSWORD')
        if not self.ID or not self.PSW:
            logging.error("Les variables d'environnement API_USER et API_PASSWORD doivent être définies.")
            raise SystemExit("Variables d'environnement manquantes.")

        logging.info("Initialisation du client XTB avec l'ID utilisateur.")
        self.ws = None
        self.exec_start = self.get_time()
# ✅ Vérifie si la connexion réussit avant de continuer
        if not self.connect():
            raise SystemExit("❌ Impossible de se connecter au serveur XTB WebSocket")

        if not self.login():
            raise SystemExit("❌ Impossible de se connecter (login XTB échoué)")

    ################ XTB ####################

    def login(self):
        login = {
            "command": "login",
            "arguments": {
                "userId": self.ID,
                "password": self.PSW
            }
        }
        login_json = json.dumps(login)
        #Sending Login Request
        result = self.send(login_json)
        result = json.loads(result)
        print("result", result)
        status = result["status"]
        if str(status) == "True":
            #Success
            return True
        else:
            #Error
            return False

    def logout(self):
        logout = {
            "command": "logout"
        }
        logout_json = json.dumps(logout)
        #Sending Logout Request
        result = self.send(logout_json)
        result = json.loads(result)
        status = result["status"]
        self.disconnect()
        if str(status) == "True":
            #Success
            return True
        else:
            #Error
            return False

    def get_AllSymbols(self):
        allsymbols = {
            "command": "getAllSymbols"
        }
        allsymbols_json = json.dumps(allsymbols)
        result = self.send(allsymbols_json)
        result = json.loads(result)
        return result

    def get_Candles(self, period, symbol, days=0, hours=0, minutes=0, qty_candles=0):
        if period == "M1":
            minutes += qty_candles
            period = 1
        elif period == "M5":
            minutes += qty_candles * 5
            period = 5
        elif period == "M15":
            minutes += qty_candles * 15
            period = 15
        elif period == "M30":
            minutes += qty_candles * 30
            period = 30
        elif period == "H1":
            minutes += qty_candles * 60
            period = 60
        elif period == "H4":
            minutes += qty_candles * 240
            period = 240
        elif period == "D1":
            minutes += qty_candles * 1440
            period = 1440
        elif period == "W1":
            minutes += qty_candles * 10080
            period = 10080
        elif period == "MN1":
            minutes += qty_candles * 43200
            period = 43200
        if qty_candles != 0:
            minutes = minutes * 2
        start = self.get_ServerTime() - self.to_milliseconds(days=days, hours=hours, minutes=minutes)
        CHART_LAST_INFO_RECORD = {
            "period": period,
            "start": start,
            "symbol": symbol
        }
        candles = {
            "command": "getChartLastRequest",
            "arguments": {
                "info": CHART_LAST_INFO_RECORD
            }
        }
        candles_json = json.dumps(candles)
        result = self.send(candles_json)
        result = json.loads(result)
        candles = []
        candle = {}
        qty = len(result["returnData"]["rateInfos"])
        candle["digits"] = result["returnData"]["digits"]
        if qty_candles == 0:
            candle["qty_candles"] = qty
        else:
            candle["qty_candles"] = qty_candles
        candles.append(candle)
        if qty_candles == 0:
            start_qty = 0
        else:
            start_qty = qty - qty_candles
        if qty == 0:
            start_qty = 0

        for i in range(start_qty, qty):
            candle = {}
            candle["datetime"] = result["returnData"]["rateInfos"][i]["ctmString"]
            candle["open"] = result["returnData"]["rateInfos"][i]["open"]
            candle["close"] = result["returnData"]["rateInfos"][i]["close"]
            candle["high"] = result["returnData"]["rateInfos"][i]["high"]
            candle["low"] = result["returnData"]["rateInfos"][i]["low"]
            candles.append(candle)
        if len(candles) == 1:
            return False
        return candles
        '''
        LIMITS:
        PERIOD_M1 --- <0-1) month, i.e. one month time
        PERIOD_M30 --- <1-7) month, six months time
        PERIOD_H4 --- <7-13) month, six months time
        PERIOD_D1 --- 13 month, and earlier on
        ##############################################
        PERIOD_M1	1	1 minute
        PERIOD_M5	5	5 minutes
        PERIOD_M15	15	15 minutes
        PERIOD_M30	30	30 minutes
        PERIOD_H1	60	60 minutes (1 hour)
        PERIOD_H4	240	240 minutes (4 hours)
        PERIOD_D1	1440	1440 minutes (1 day)
        PERIOD_W1	10080	10080 minutes (1 week)
        PERIOD_MN1	43200	43200 minutes (30 days)
        ##############################################
        close   Value of close price (shift from open price)
        ctm     Candle start time in CET / CEST time zone (see Daylight Saving Time)
        ctmString     String representation of the 'ctm' field
        high    Highest value in the given period (shift from open price)
        low     Lowest value in the given period (shift from open price)
        open    Open price (in base currency * 10 to the power of digits)
        vol     Volume in lots
        '''

    def get_CandlesRange(self, period, symbol, start=0, end=0, days=0, qty_candles=0):
        if period == "M1":
            period = 1
        elif period == "M5":
            period = 5
        elif period == "M15":
            period = 15
        elif period == "M30":
            period = 30
        elif period == "H1":
            period = 60
        elif period == "H4":
            period = 240
        elif period == "D1":
            period = 1440
        elif period == "W1":
            period = 10080
        elif period == "MN1":
            period = 43200

        if end == 0:
            end = self.get_time()
            end = end.strftime('%m/%d/%Y %H:%M:%S')
            if start == 0:
                if qty_candles == 0:
                    temp = datetime.strptime(end, '%m/%d/%Y %H:%M:%S')
                    start = temp - timedelta(days=days)
                    start = start.strftime("%m/%d/%Y %H:%M:%S")
                else:
                    start = datetime.strptime(end, '%m/%d/%Y %H:%M:%S')
                    minutes = period * qty_candles
                    start = start - timedelta(minutes=minutes)
                    start = start.strftime("%m/%d/%Y %H:%M:%S")

        start = self.time_conversion(start)
        end = self.time_conversion(end)

        CHART_RANGE_INFO_RECORD = {
            "end": end,
            "period": period,
            "start": start,
            "symbol": symbol,
            "ticks": 0
        }
        candles = {
            "command": "getChartRangeRequest",
            "arguments": {
                "info": CHART_RANGE_INFO_RECORD
            }
        }
        candles_json = json.dumps(candles)
        result = self.send(candles_json)
        result = json.loads(result)
        candles = []
        candle = {}
        qty = len(result["returnData"]["rateInfos"])
        candle["digits"] = result["returnData"]["digits"]
        if qty_candles == 0:
            candle["qty_candles"] = qty
        else:
            candle["qty_candles"] = qty_candles
        candles.append(candle)
        if qty_candles == 0:
            start_qty = 0
        else:
            start_qty = qty - qty_candles
        if qty == 0:
            start_qty = 0
        for i in range(start_qty, qty):
            candle = {}
            candle["datetime"] = str(result["returnData"]["rateInfos"][i]["ctmString"])
            candle["open"] = result["returnData"]["rateInfos"][i]["open"]
            candle["close"] = result["returnData"]["rateInfos"][i]["close"]
            candle["high"] = result["returnData"]["rateInfos"][i]["high"]
            candle["low"] = result["returnData"]["rateInfos"][i]["low"]
            candles.append(candle)
        if len(candles) == 1:
            return False
        return candles
        '''
        LIMITS:
        PERIOD_M1 --- <0-1) month, i.e. one month time
        PERIOD_M30 --- <1-7) month, six months time
        PERIOD_H4 --- <7-13) month, six months time
        PERIOD_D1 --- 13 month, and earlier on
        ##############################################
        PERIOD_M1	1	1 minute
        PERIOD_M5	5	5 minutes
        PERIOD_M15	15	15 minutes
        PERIOD_M30	30	30 minutes
        PERIOD_H1	60	60 minutes (1 hour)
        PERIOD_H4	240	240 minutes (4 hours)
        PERIOD_D1	1440	1440 minutes (1 day)
        PERIOD_W1	10080	10080 minutes (1 week)
        PERIOD_MN1	43200	43200 minutes (30 days)
        ##############################################
        close   Value of close price (shift from open price)
        ctm     Candle start time in CET / CEST time zone (see Daylight Saving Time)
        ctmString     String representation of the 'ctm' field
        high    Highest value in the given period (shift from open price)
        low     Lowest value in the given period (shift from open price)
        open    Open price (in base currency * 10 to the power of digits)
        vol     Volume in lots
        '''

    def get_ServerTime(self):
        time = {
            "command": "getServerTime"
        }
        time_json = json.dumps(time)
        result = self.send(time_json)
        result = json.loads(result)
        print(result)
        time = result["returnData"]["time"]
        return time

    def get_Balance(self):
        balance = {
            "command": "getMarginLevel"
        }
        balance_json = json.dumps(balance)
        result = self.send(balance_json)
        result = json.loads(result)
        balance = result["returnData"]["balance"]
        return balance

    def get_Margin(self, symbol, volume):
        margin = {
            "command": "getMarginTrade",
            "arguments": {
                "symbol": symbol,
                "volume": volume
            }
        }
        margin_json = json.dumps(margin)
        result = self.send(margin_json)
        result = json.loads(result)
        margin = result["returnData"]["margin"]
        return margin

    def get_Profit(self, open_price, close_price, transaction_type, symbol, volume):
        if transaction_type == 1:
            #buy
            cmd = 0
        else:
            #sell
            cmd = 1
        profit = {
            "command": "getProfitCalculation",
            "arguments": {
                "closePrice": close_price,
                "cmd": cmd,
                "openPrice": open_price,
                "symbol": symbol,
                "volume": volume
            }
        }
        profit_json = json.dumps(profit)
        result = self.send(profit_json)
        result = json.loads(result)
        profit = result["returnData"]["profit"]
        return profit

    def get_Symbol(self, symbol):
        symbol = {
            "command": "getSymbol",
            "arguments": {
                "symbol": symbol
            }
        }
        symbol_json = json.dumps(symbol)
        result = self.send(symbol_json)
        result = json.loads(result)
        symbol = result["returnData"]
        return symbol

    def make_Trade(self, symbol, cmd, volume, comment="", order=0, sl=0, tp=0, days=0, hours=0, minutes=0):
        price = self.get_Candles("M1", symbol, qty_candles=1)
        price = price[1]["open"] + price[1]["close"]

        delay = self.to_milliseconds(days=days, hours=hours, minutes=minutes)
        if delay == 0:
            expiration = self.get_ServerTime() + self.to_milliseconds(minutes=1)
        else:
            expiration = self.get_ServerTime() + delay

        TRADE_TRANS_INFO = {
            "cmd": cmd,
            "customComment": comment,
            "expiration": expiration,
            "offset": -1,
            "order": order,
            "price": price,
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
            "type": type_mapping_inverse.get("order_open"),
            "volume": volume
        }
        trade = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": TRADE_TRANS_INFO
            }
        }
        trade_json = json.dumps(trade)
        result = self.send(trade_json)
        result = json.loads(result)
        if result["status"] == True:
            #success
            logging.info(f"price: result {price} ")
            return True, result["returnData"]["order"]
        else:
            #error
            return False, 0
        """
        format TRADE_TRANS_INFO:
        cmd	        Number	            Operation code
        customComment	String	            The value the customer may provide in order to retrieve it later.
        expiration	Time	            Pending order expiration time
        offset	        Number	            Trailing offset
        order	        Number	            0 or position number for closing/modifications
        price	        Floating number	    Trade price
        sl	        Floating number	    Stop loss
        symbol	        String	            Trade symbol
        tp	        Floating number	    Take profit
        type	        Number	            Trade transaction type
        volume	        Floating number	    Trade volume

        values cmd:
        BUY	        0	buy
        SELL	        1	sell
        BUY_LIMIT	2	buy limit
        SELL_LIMIT	3	sell limit
        BUY_STOP	4	buy stop
        SELL_STOP	5	sell stop
        BALANCE	        6	Read only. Used in getTradesHistory  for manager's deposit/withdrawal operations (profit>0 for deposit, profit<0 for withdrawal).
        CREDIT	        7	Read only

        values transaction_type:
        OPEN	    0	    order open, used for opening orders
        PENDING	    1	    order pending, only used in the streaming getTrades  command
        CLOSE	    2	    order close
        MODIFY	    3	    order modify, only used in the tradeTransaction  command
        DELETE	    4	    order delete, only used in the tradeTransaction  command
        """

    def close_Trade(self, order_id, symbol, cmd, volume, price, comment=""):
        TRADE_TRANS_INFO = {
            "cmd": 0,
            "customComment": comment,
            "expiration": 0,
            "order": int(order_id),
            "price": price,
            "symbol": symbol,
            "sl": 0,
            "tp": 0,
            "type": type_mapping_inverse.get("order_close"),
            "volume": volume
        }
        print(f"Tentative de fermeture de la position avec TRADE_TRANS_INFO: {TRADE_TRANS_INFO}")
        trade = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": TRADE_TRANS_INFO
            }
        }
        trade_json = json.dumps(trade)
        result = self.send(trade_json)
        result = json.loads(result)
        print(result)
        if result["status"] == True:
            #success
            return True, result["returnData"]["order"]
        else:
            #error
            return False, 0

    def check_Trade(self, order):
        trade = {
            "command": "tradeTransactionStatus",
            "arguments": {
                "order": order
            }
        }
        trade_json = json.dumps(trade)
        result = self.send(trade_json)
        result = json.loads(result)
        logging.info(f"Tcheck_Trade: result {result} ")
        status = result["returnData"]["requestStatus"]
        return status

    '''
    ERROR 	0	error
    PENDING	1	pending
    ACCEPTED	3	The transaction has been executed successfully
    REJECTED	4	The transaction has been rejected
    '''

    def get_position_by_order_id(self, order_id):
        # Préparer la requête pour obtenir toutes les positions ouvertes
        trades_request = {
            "command": "getTrades",
            "arguments": {
                "openedOnly": True  # Récupérer uniquement les trades ouverts
            }
        }

        # Envoyer la requête
        trades_json = json.dumps(trades_request)
        response = self.send(trades_json)
        trades = json.loads(response)
        # Vérifier si la requête a réussi
        if trades["status"] == True:
            # Parcourir toutes les positions ouvertes pour trouver celle avec l'ID d'ordre correspondant
            for trade in trades["returnData"]:
                if trade["order2"] == order_id:
                    # Trouvé, retourner les détails de la position
                    return trade

            # Si l'ID d'ordre n'a pas été trouvé parmi les positions ouvertes
            print(f"Aucune position trouvée avec l'ID d'ordre {order_id}.")
            return None
        else:
            # Si la requête a échoué
            print(f"Erreur lors de la récupération des positions : {trades['errorDescr']}")
            return None

    def get_position(self, position_id):
        # Préparer la requête pour obtenir toutes les positions ouvertes
        trades_request = {
            "command": "getTrades",
            "arguments": {
                "openedOnly": False  # Récupérer uniquement les trades ouverts
            }
        }

        # Envoyer la requête
        trades_json = json.dumps(trades_request)
        response = self.send(trades_json)
        trades = json.loads(response)
        print(trades)
        # Vérifier si la requête a réussi
        if trades["status"] == True:
            # Parcourir toutes les positions ouvertes pour trouver celle avec l'ID d'ordre correspondant
            for trade in trades["returnData"]:
                if trade["position"] == position_id:
                    # Trouvé, retourner les détails de la position
                    return trade

            # Si l'ID d'ordre n'a pas été trouvé parmi les positions ouvertes
            print(f"Aucune position trouvée avec l'ID d'ordre {position_id}.")
            return None
        else:
            # Si la requête a échoué
            print(f"Erreur lors de la récupération des positions : {trades['errorDescr']}")
            return None
    def get_today_history(self):
        # Obtenir le temps du serveur actuel
        server_time = self.get_ServerTime()

        # Convertir le temps du serveur en format datetime
        current_date = datetime.fromtimestamp(server_time / 1000)  # Diviser par 1000 pour convertir des millisecondes aux secondes

        # Obtenir le début de la journée (minuit) en UTC
        start_of_day = current_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # Convertir les temps en millisecondes pour l'API XTB
        start_timestamp = int(start_of_day.timestamp() * 1000)  # Convertir en millisecondes
        end_timestamp = server_time  # Fin à l'heure actuelle

        # Utiliser la méthode get_History avec les timestamps du début et de la fin de la journée
        history = self.get_History(start=start_timestamp, end=end_timestamp)

        return history
    def get_History(self, start=0, end=0, days=0, hours=0, minutes=0):
        # Vérifie si 'start' et 'end' sont déjà des timestamps (entiers)
        if isinstance(start, str):
            start = self.time_conversion(start)
        if isinstance(end, str):
            end = self.time_conversion(end)

        if days != 0 or hours != 0 or minutes != 0:
            if end == 0:
                end = self.get_ServerTime()
            start = end - self.to_milliseconds(days=days, hours=hours, minutes=minutes)

        history = {
            "command": "getTradesHistory",
            "arguments": {
                "end": end,
                "start": start
            }
        }
        history_json = json.dumps(history)
        result = self.send(history_json)
        result = json.loads(result)
        history = result["returnData"]
        return history

    def ping(self):
        ping = {
            "command": "ping"
        }
        ping_json = json.dumps(ping)
        result = self.send(ping_json)
        result = json.loads(result)
        return result["status"]

    ################ TIME/DATE/CONVERSIONS ####################

    def get_time(self):
        time = datetime.today().strftime('%m/%d/%Y %H:%M:%S%f')
        time = datetime.strptime(time, '%m/%d/%Y %H:%M:%S%f')
        return time

    def to_milliseconds(self, days=0, hours=0, minutes=0):
        milliseconds = (days * 24 * 60 * 60 * 1000) + (hours * 60 * 60 * 1000) + (minutes * 60 * 1000)
        return milliseconds

    def time_conversion(self, date):
        start = "01/01/1970 00:00:00"
        start = datetime.strptime(start, '%m/%d/%Y %H:%M:%S')
        date = datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
        final_date = date - start
        temp = str(final_date)
        temp1, temp2 = temp.split(", ")
        hours, minutes, seconds = temp2.split(":")
        days = final_date.days
        days = int(days)
        hours = int(hours)
        hours += 2
        minutes = int(minutes)
        seconds = int(seconds)
        time = (days * 24 * 60 * 60 * 1000) + (hours * 60 * 60 * 1000) + (minutes * 60 * 1000) + (seconds * 1000)
        return time

    ################ CHECKS ####################

    def is_on(self):
        temp1 = self.exec_start
        temp2 = self.get_time()
        temp = temp2 - temp1
        temp = temp.total_seconds()
        temp = float(temp)
        if temp >= 8.0:
            self.connect()
        self.exec_start = self.get_time()

    def is_open(self, symbol):
        candles = self.get_Candles("M1", symbol, qty_candles=1)
        if len(candles) == 1:
            return False
        else:
            return True

    ################ WEBSOCKETS ####################

    def connect(self):
        try:
            self.ws = websocket.create_connection("wss://ws.xtb.com/demo")
            #Success
            return True
        except:
            #Error
            return False

    def disconnect(self):
        try:
            self.ws.close()
            #Success
            return True
        except:
            return False

    def send(self, msg):

        try:
            self.is_on()  # Vérifie la connexion
            self.ws.send(msg)  # Envoi du message
            result = self.ws.recv()  # Réception de la réponse
            return result + "\n"
        except WebSocketConnectionClosedException:
            logging.warning("Connexion WebSocket fermée. Tentative de reconnexion...")
            self.connect()  # Reconnexion à la WebSocket
            try:
                self.ws.send(msg)  # Réessayer d'envoyer le message
                result = self.ws.recv()  # Réception de la réponse
                return result + "\n"
            except Exception as e:
                logging.error(f"Erreur lors de la nouvelle tentative d'envoi après reconnexion: {e}")
                raise e
